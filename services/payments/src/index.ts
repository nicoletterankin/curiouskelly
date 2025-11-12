import Stripe from "stripe";
import { Queue, Worker, QueueEvents, Job } from "bullmq";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { notificationsService } from "@acme/service-notifications";

const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
  apiVersion: "2024-06-20",
});

export class PaymentsService {
  private readonly dunningQueue: Queue | null;
  private readonly queueEvents: QueueEvents | null;

  private dunningWorker?: Worker;

  constructor() {
    let queue: Queue | null = null;
    try {
      queue = new Queue("payments.dunning", {
        connection: { url: env.REDIS_URL },
      });
    } catch (error) {
      logger.warn({ error }, "PaymentsService.queue_disabled");
    }
    this.dunningQueue = queue;

    let events: QueueEvents | null = null;
    try {
      events = new QueueEvents("payments.dunning", {
        connection: { url: env.REDIS_URL },
      });
      events.on("failed", ({ jobId, failedReason }) =>
        logger.error({ jobId, failedReason }, "PaymentsService.dunning.failed"),
      );
      events.on("completed", ({ jobId }) =>
        logger.info({ jobId }, "PaymentsService.dunning.completed"),
      );
    } catch (error) {
      logger.warn({ error }, "PaymentsService.queue_events_disabled");
    }
    this.queueEvents = events;
  }

  async startDunningWorker() {
    if (this.dunningWorker || !this.dunningQueue) {
      return;
    }

    try {
      this.dunningWorker = new Worker(
        "payments.dunning",
        async (job: Job<{ customerId: string; invoiceId: string }>) => {
          await this.processDunningJob(job.data.customerId, job.data.invoiceId);
        },
        {
          connection: { url: env.REDIS_URL },
        },
      );
      this.dunningWorker.on("error", (error) =>
        logger.error({ error }, "PaymentsService.dunning.worker_error"),
      );
    } catch (error) {
      logger.warn({ error }, "PaymentsService.dunning.worker_disabled");
    }
  }

  async createCheckoutSession(userId: string, priceId: string) {
    const customer = await this.ensureCustomer(userId);
    const session = await stripe.checkout.sessions.create({
      customer,
      mode: "subscription",
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${env.LIVEKIT_HOST}/payments/success`,
      cancel_url: `${env.LIVEKIT_HOST}/payments/cancel`,
    });
    return session;
  }

  async handleWebhook(signature: string, payload: Buffer) {
    const event = stripe.webhooks.constructEvent(payload, signature, env.STRIPE_WEBHOOK_SECRET);
    switch (event.type) {
      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted":
        await this.syncSubscription(event.data.object as Stripe.Subscription);
        break;
      case "invoice.payment_failed":
        await this.scheduleDunning(event.data.object as Stripe.Invoice);
        break;
      default:
        logger.debug({ type: event.type }, "PaymentsService.webhook.unhandled");
    }
  }

  private async ensureCustomer(userId: string) {
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user) throw new Error("user not found");

    const existing = await prisma.subscription.findFirst({
      where: { userId },
      select: { stripeCustomerId: true },
    });
    if (existing?.stripeCustomerId) {
      return existing.stripeCustomerId;
    }

    const customer = await stripe.customers.create({
      email: user.email,
      name: user.displayName,
    });

    await prisma.subscription.create({
      data: {
        userId,
        plan: {
          connect: {
            code: "FREE",
          },
        },
        stripeCustomerId: customer.id,
        stripeSubscriptionId: "pending",
        status: "INCOMPLETE",
        currentPeriodEnd: new Date(),
      },
    });

    return customer.id;
  }

  private async syncSubscription(subscription: Stripe.Subscription) {
    const planCode = subscription.items.data[0]?.price.nickname ?? "STANDARD";
    const plan = await prisma.plan.findFirst({ where: { code: planCode } });
    if (!plan) {
      logger.error({ planCode }, "PaymentsService.syncSubscription.planMissing");
      return;
    }

    await prisma.subscription.updateMany({
      where: { stripeSubscriptionId: subscription.id },
      data: {
        planId: plan.id,
        status: subscription.status.toUpperCase() as any,
        currentPeriodEnd: new Date(subscription.current_period_end * 1000),
        cancelAtPeriodEnd: Boolean(subscription.cancel_at_period_end),
      },
    });
  }

  private async scheduleDunning(invoice: Stripe.Invoice) {
    if (!invoice.customer) return;
    if (!this.dunningQueue) {
      logger.warn("PaymentsService.scheduleDunning.queue_unavailable");
      return;
    }

    await this.dunningQueue.add(
      "invoice",
      {
        customerId: invoice.customer,
        invoiceId: invoice.id,
      },
      {
        attempts: 5,
        backoff: {
          type: "exponential",
          delay: 60_000,
        },
      },
    );
  }

  private async processDunningJob(customerId: string | Stripe.Customer, invoiceId: string) {
    const invoice = await stripe.invoices.retrieve(invoiceId, {
      expand: ["subscription", "customer"],
    });

    if (invoice.status === "paid") {
      return;
    }

    try {
      const retry = await stripe.invoices.pay(invoiceId);
      if (retry.status === "paid") {
        await this.markSubscriptionStatus(retry.subscription as string, "ACTIVE");
        await notificationsService.enqueue({
          userId: await this.resolveUserId(customerId),
          channel: "EMAIL",
          template: "payment_retry_success",
          payload: {
            invoiceId,
            amount: retry.amount_due,
            currency: retry.currency,
          },
        });
        return;
      }
      throw new Error(`Invoice retry status: ${retry.status}`);
    } catch (error) {
      const userId = await this.resolveUserId(customerId);
      await this.markSubscriptionStatus(invoice.subscription as string, "PAST_DUE");
      await notificationsService.enqueue({
        userId,
        channel: "EMAIL",
        template: "payment_retry_failed",
        payload: {
          invoiceId,
          currency: invoice.currency,
          amount: invoice.amount_due,
        },
      });
      throw error;
    }
  }

  private async resolveUserId(customerId: string | Stripe.Customer): Promise<string> {
    const id = typeof customerId === "string" ? customerId : customerId.id;
    const subscription = await prisma.subscription.findFirst({
      where: {
        stripeCustomerId: id,
      },
    });
    if (!subscription) {
      throw new Error(`Subscription not found for customer ${id}`);
    }
    return subscription.userId;
  }

  private async markSubscriptionStatus(subscriptionId: string, status: string) {
    await prisma.subscription.updateMany({
      where: { stripeSubscriptionId: subscriptionId },
      data: {
        status,
      },
    });
  }
}

export const paymentsService = new PaymentsService();


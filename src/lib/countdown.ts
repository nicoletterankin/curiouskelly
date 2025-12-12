export interface CountdownUnits {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
}

export interface CountdownState extends CountdownUnits {
  active: boolean;
}

export function calculateCountdown(targetDate: Date): CountdownState {
  const now = new Date();
  const diff = targetDate.getTime() - now.getTime();

  if (Number.isNaN(targetDate.getTime()) || diff <= 0) {
    return {
      active: false,
      days: 0,
      hours: 0,
      minutes: 0,
      seconds: 0
    };
  }

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  return {
    active: true,
    days,
    hours: hours % 24,
    minutes: minutes % 60,
    seconds: seconds % 60
  };
}

export function formatCountdownValue(value: number): string {
  return value.toString().padStart(2, '0');
}

export function startCountdown(
  targetIsoDate: string,
  callback: (state: CountdownState) => void
): () => void {
  const targetDate = new Date(targetIsoDate);
  const tick = () => {
    callback(calculateCountdown(targetDate));
  };

  tick();
  const interval = setInterval(tick, 1000);
  return () => clearInterval(interval);
}













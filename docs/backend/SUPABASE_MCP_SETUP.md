# Supabase MCP Setup

To enable deep integration between Cursor and your Supabase project (allowing Cursor to read schema, query data, and understand your database context), you need to configure the Model Context Protocol (MCP) server.

## Configuration

Create or update the file `.cursor/mcp.json` in the root of your workspace with the following configuration:

```json
{
  "mcpServers": {
    "supabase": {
      "url": "https://mcp.supabase.com/mcp?project_ref=tvjalxxsyryjphkforjv"
    }
  }
}
```

## Benefits
- **Schema Awareness**: The AI assistant can see your live table definitions.
- **Data Inspection**: You can ask questions like "Show me the latest lesson in core_lessons".
- **Contextual Coding**: When writing backend code, the assistant knows the exact field names and types.

## Verification
After adding the file, restart Cursor or reload the window. You should see "Supabase" active in the Cursor AI pane or settings.










































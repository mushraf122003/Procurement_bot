# Procurement Bot

## Agent tools

`scr_code/procurement_graph.py` gives the Gemini agent three tool choices:

- `supplier_contract`: retrieves supplier contract details from MongoDB Atlas Vector Search.
- `purchase_order`: generates and runs a read-only query against allowed purchase-order tables in MySQL.
- `invoice`: generates and runs a read-only query against allowed invoice tables in MySQL.

Configure `MYSQL_URI`, `MYSQL_PURCHASE_ORDER_TABLES`, and `MYSQL_INVOICE_TABLES` using `.env.example`. Each MySQL tool provides Gemini with the schema and one sample row from every allowed table. It accepts only one read-only `SELECT` or `WITH` statement and rejects writes, comments, multiple statements, and tables outside its own allowlist.

Run a conversation using the same `--thread-id` to retain LangGraph checkpoint memory for that conversation:

```powershell
python -m scr_code.procurement_graph --thread-id vendor-review-001 "Show purchase orders for ACME."
```

"""
PostgreSQL Setup & Connection Utility for ApproxMAC Employee System
Usage:
    python setup_postgres.py [--host localhost] [--port 5432] [--user postgres] [--password postgres] [--dbname approxmac_db]
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL Database for ApproxMAC")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"), help="PostgreSQL host")
    parser.add_argument("--port", default=os.environ.get("PGPORT", "5432"), help="PostgreSQL port")
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"), help="PostgreSQL user")
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", "postgres"), help="PostgreSQL password")
    parser.add_argument("--dbname", default=os.environ.get("PGDATABASE", "approxmac_db"), help="Database name")
    args = parser.parse_args()

    db_url = f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/{args.dbname}"
    admin_url = f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/postgres"

    print("=" * 60)
    print(" ApproxMAC PostgreSQL Database Initializer")
    print("=" * 60)
    print(f"Target DB: {args.dbname} on {args.host}:{args.port} (user: {args.user})")

    try:
        from sqlalchemy import create_engine, text
        import psycopg2
    except ImportError:
        print("Error: Missing psycopg2-binary or sqlalchemy. Install with:")
        print("  pip install psycopg2-binary sqlalchemy")
        sys.exit(1)

    # 1. Test server connection via default 'postgres' database
    print(f"\n[1/3] Connecting to PostgreSQL server at {args.host}:{args.port}...")
    try:
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            # Check if target db exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{args.dbname}'"))
            exists = result.scalar() is not None
            if not exists:
                print(f"[2/3] Database '{args.dbname}' does not exist. Creating database...")
                conn.execute(text(f"CREATE DATABASE {args.dbname}"))
                print(f"      Database '{args.dbname}' created successfully.")
            else:
                print(f"[2/3] Database '{args.dbname}' already exists.")
    except Exception as e:
        print(f"\n[Warning] Could not connect to PostgreSQL server: {e}")
        print("\nPlease verify:")
        print("  1. PostgreSQL service is running on your machine.")
        print(f"  2. Credentials ({args.user} / password) are correct.")
        print("  3. Or start the server with your custom connection string:")
        print(f"     set DATABASE_URL=postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.dbname}")
        print("     python server.py")
        print("\nNote: ApproxMAC automatically falls back to SQLite so your app continues to function seamlessly without downtime!")
        sys.exit(1)

    # 3. Create tables in target database
    print(f"\n[3/3] Initializing employee schema in '{args.dbname}'...")
    try:
        os.environ["DATABASE_URL"] = db_url
        import database
        database.Base.metadata.create_all(bind=database.engine)
        print("      Schema initialized successfully! Employee table is ready.")
        print("\n" + "=" * 60)
        print(" SUCCESS: ApproxMAC is configured to run with PostgreSQL!")
        print(f" Connection URL: {db_url}")
        print("=" * 60)
    except Exception as e:
        print(f"Error creating schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

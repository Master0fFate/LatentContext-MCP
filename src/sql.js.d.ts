declare module "sql.js" {
    export interface Database {
        run(sql: string, params?: unknown[]): Database;
        exec(sql: string): QueryExecResult[];
        prepare(sql: string): Statement;
        export(): Uint8Array;
        close(): void;
    }

    export interface Statement {
        bind(params?: unknown[]): boolean;
        step(): boolean;
        getAsObject(params?: unknown): Record<string, unknown>;
        free(): boolean;
        reset(): void;
    }

    export interface QueryExecResult {
        columns: string[];
        values: unknown[][];
    }

    export interface SqlJsStatic {
        Database: new (data?: ArrayLike<number>) => Database;
    }

    export default function initSqlJs(config?: Record<string, unknown>): Promise<SqlJsStatic>;
}

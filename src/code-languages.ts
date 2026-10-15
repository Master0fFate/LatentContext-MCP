import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const SYMBOL_KINDS = [
    "function", "method", "class", "struct", "interface", "trait", "enum", "type", "module", "impl", "macro", "constant",
] as const;
export type SymbolKind = (typeof SYMBOL_KINDS)[number];

export const REF_KINDS = ["type", "import", "inherit", "call"] as const;
export type RefKind = (typeof REF_KINDS)[number];

export const CONTAINER_KINDS: ReadonlySet<SymbolKind> = new Set(["class", "struct", "interface", "trait", "impl", "enum"]);

const vscodeGrammars = join(dirname(createRequire(import.meta.url).resolve("@vscode/tree-sitter-wasm/package.json")), "wasm");
const bundledGrammars = fileURLToPath(new URL("../grammars/", import.meta.url));

const vscode = (name: string): string => join(vscodeGrammars, `tree-sitter-${name}.wasm`);
const bundled = (name: string): string => join(bundledGrammars, `tree-sitter-${name}.wasm`);

const JS_COMMON = `
(function_declaration name: (identifier) @name) @def.function
(generator_function_declaration name: (identifier) @name) @def.function
(method_definition name: [(property_identifier) (private_property_identifier)] @name) @def.method
(lexical_declaration (variable_declarator name: (identifier) @name value: [(arrow_function) (function_expression) (generator_function)])) @def.function
(variable_declaration (variable_declarator name: (identifier) @name value: [(arrow_function) (function_expression) (generator_function)])) @def.function
(pair key: (property_identifier) @name value: [(arrow_function) (function_expression)]) @def.method
(assignment_expression left: (member_expression property: (property_identifier) @name) right: [(arrow_function) (function_expression)]) @def.method
(program (lexical_declaration (variable_declarator name: (identifier) @name)) @def.constant)
(program (export_statement (lexical_declaration (variable_declarator name: (identifier) @name)) @def.constant))
(call_expression function: (identifier) @ref.call)
(call_expression function: (member_expression property: [(property_identifier) (private_property_identifier)] @ref.call))
(new_expression constructor: (identifier) @ref.call)
(new_expression constructor: (member_expression property: (property_identifier) @ref.call))
(decorator (identifier) @ref.call)
(import_specifier name: (identifier) @ref.import)
(import_clause (identifier) @ref.import)
`;

const JSX = `
(jsx_opening_element name: (identifier) @ref.call)
(jsx_self_closing_element name: (identifier) @ref.call)
(jsx_opening_element name: (member_expression property: (property_identifier) @ref.call))
(jsx_self_closing_element name: (member_expression property: (property_identifier) @ref.call))
`;

const TS = `${JS_COMMON}
(class_declaration name: (type_identifier) @name) @def.class
(abstract_class_declaration name: (type_identifier) @name) @def.class
(class name: (type_identifier) @name) @def.class
(interface_declaration name: (type_identifier) @name) @def.interface
(type_alias_declaration name: (type_identifier) @name) @def.type
(enum_declaration name: (identifier) @name) @def.enum
(internal_module name: (identifier) @name) @def.module
(function_signature name: (identifier) @name) @def.function
(method_signature name: (property_identifier) @name) @def.method
(abstract_method_signature name: (property_identifier) @name) @def.method
(public_field_definition name: (property_identifier) @name value: [(arrow_function) (function_expression)]) @def.method
(extends_clause value: (identifier) @ref.inherit)
(extends_clause value: (member_expression property: (property_identifier) @ref.inherit))
(implements_clause (type_identifier) @ref.inherit)
(implements_clause (generic_type name: (type_identifier) @ref.inherit))
(extends_type_clause type: (type_identifier) @ref.inherit)
(extends_type_clause type: (generic_type name: (type_identifier) @ref.inherit))
(type_identifier) @ref.type
`;

const JS = `${JS_COMMON}${JSX}
(class_declaration name: (identifier) @name) @def.class
(class name: (identifier) @name) @def.class
(field_definition property: (property_identifier) @name value: [(arrow_function) (function_expression)]) @def.method
(class_heritage (identifier) @ref.inherit)
(class_heritage (member_expression property: (property_identifier) @ref.inherit))
`;

const PYTHON = `
(class_definition name: (identifier) @name) @def.class
(function_definition name: (identifier) @name) @def.function
(module (expression_statement (assignment left: (identifier) @name)) @def.constant)
(call function: (identifier) @ref.call)
(call function: (attribute attribute: (identifier) @ref.call))
(decorator (identifier) @ref.call)
(class_definition superclasses: (argument_list (identifier) @ref.inherit))
(class_definition superclasses: (argument_list (attribute attribute: (identifier) @ref.inherit)))
(import_from_statement name: (dotted_name (identifier) @ref.import))
(import_from_statement name: (aliased_import name: (dotted_name (identifier) @ref.import)))
(type (identifier) @ref.type)
`;

const RUST = `
(struct_item name: (type_identifier) @name) @def.struct
(union_item name: (type_identifier) @name) @def.struct
(enum_item name: (type_identifier) @name) @def.enum
(type_item name: (type_identifier) @name) @def.type
(trait_item name: (type_identifier) @name) @def.trait
(function_item name: (identifier) @name) @def.function
(function_signature_item name: (identifier) @name) @def.function
(mod_item name: (identifier) @name) @def.module
(macro_definition name: (identifier) @name) @def.macro
(const_item name: (identifier) @name) @def.constant
(static_item name: (identifier) @name) @def.constant
(impl_item type: [(type_identifier) @name (generic_type type: (type_identifier) @name) (scoped_type_identifier name: (type_identifier) @name)]) @def.impl
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression field: (field_identifier) @ref.call))
(call_expression function: (scoped_identifier name: (identifier) @ref.call))
(call_expression function: (generic_function function: [(identifier) @ref.call (scoped_identifier name: (identifier) @ref.call) (field_expression field: (field_identifier) @ref.call)]))
(macro_invocation macro: (identifier) @ref.call)
(struct_expression name: (type_identifier) @ref.call)
(impl_item trait: [(type_identifier) @ref.inherit (generic_type type: (type_identifier) @ref.inherit) (scoped_type_identifier name: (type_identifier) @ref.inherit)])
(use_declaration argument: (scoped_identifier name: (identifier) @ref.import))
(use_list (identifier) @ref.import)
(type_identifier) @ref.type
`;

const GO = `
(function_declaration name: (identifier) @name) @def.function
(method_declaration receiver: (parameter_list (parameter_declaration type: [(type_identifier) @owner (pointer_type (type_identifier) @owner) (generic_type type: (type_identifier) @owner) (pointer_type (generic_type type: (type_identifier) @owner))])) name: (field_identifier) @name) @def.method
(method_declaration name: (field_identifier) @name) @def.method
(type_spec name: (type_identifier) @name type: (struct_type)) @def.struct
(type_spec name: (type_identifier) @name type: (interface_type)) @def.interface
(type_spec name: (type_identifier) @name) @def.type
(type_alias name: (type_identifier) @name) @def.type
(const_spec name: (identifier) @name) @def.constant
(call_expression function: (identifier) @ref.call)
(call_expression function: (selector_expression field: (field_identifier) @ref.call))
(composite_literal type: (type_identifier) @ref.call)
(type_identifier) @ref.type
`;

const JAVA = `
(class_declaration name: (identifier) @name) @def.class
(interface_declaration name: (identifier) @name) @def.interface
(enum_declaration name: (identifier) @name) @def.enum
(record_declaration name: (identifier) @name) @def.class
(annotation_type_declaration name: (identifier) @name) @def.interface
(method_declaration name: (identifier) @name) @def.method
(constructor_declaration name: (identifier) @name) @def.method
(method_invocation name: (identifier) @ref.call)
(object_creation_expression type: (type_identifier) @ref.call)
(object_creation_expression type: (generic_type (type_identifier) @ref.call))
(superclass (type_identifier) @ref.inherit)
(super_interfaces (type_list (type_identifier) @ref.inherit))
(import_declaration (scoped_identifier name: (identifier) @ref.import))
(type_identifier) @ref.type
`;

const CSHARP = `
(class_declaration name: (identifier) @name) @def.class
(struct_declaration name: (identifier) @name) @def.struct
(interface_declaration name: (identifier) @name) @def.interface
(enum_declaration name: (identifier) @name) @def.enum
(record_declaration name: (identifier) @name) @def.class
(delegate_declaration name: (identifier) @name) @def.type
(method_declaration name: (identifier) @name) @def.method
(constructor_declaration name: (identifier) @name) @def.method
(local_function_statement name: (identifier) @name) @def.function
(namespace_declaration name: [(identifier) (qualified_name)] @name) @def.module
(file_scoped_namespace_declaration name: [(identifier) (qualified_name)] @name) @def.module
(invocation_expression function: (identifier) @ref.call)
(invocation_expression function: (member_access_expression name: (identifier) @ref.call))
(invocation_expression function: (generic_name (identifier) @ref.call))
(object_creation_expression type: (identifier) @ref.call)
(object_creation_expression type: (generic_name (identifier) @ref.call))
(base_list (identifier) @ref.inherit)
(base_list (generic_name (identifier) @ref.inherit))
`;

const C_FAMILY_COMMON = `
(struct_specifier name: (type_identifier) @name body: (_)) @def.struct
(union_specifier name: (type_identifier) @name body: (_)) @def.struct
(enum_specifier name: (type_identifier) @name body: (_)) @def.enum
(type_definition declarator: (type_identifier) @name) @def.type
(preproc_function_def name: (identifier) @name) @def.macro
(preproc_def name: (identifier) @name) @def.constant
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression field: (field_identifier) @ref.call))
(type_identifier) @ref.type
`;

const C = `${C_FAMILY_COMMON}
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def.function
(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name))) @def.function
(function_definition declarator: (pointer_declarator declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name)))) @def.function
`;

const CPP_NAME = "[(identifier) (field_identifier) (destructor_name) (operator_name)] @name";
const CPP_QUALIFIED = "(qualified_identifier scope: (_) @owner name: [(identifier) (destructor_name) (operator_name)] @name)";
const cppFunction = (declarator: string, kind: string): string => [
    `(function_definition declarator: (function_declarator declarator: ${declarator})) @def.${kind}`,
    `(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: ${declarator}))) @def.${kind}`,
    `(function_definition declarator: (reference_declarator (function_declarator declarator: ${declarator}))) @def.${kind}`,
].join("\n");

const CPP = `${C_FAMILY_COMMON}
${cppFunction(CPP_NAME, "function")}
${cppFunction(CPP_QUALIFIED, "method")}
(class_specifier name: (type_identifier) @name body: (_)) @def.class
(namespace_definition name: (namespace_identifier) @name) @def.module
(alias_declaration name: (type_identifier) @name) @def.type
(call_expression function: (qualified_identifier name: (identifier) @ref.call))
(call_expression function: (template_function name: (identifier) @ref.call))
(new_expression type: (type_identifier) @ref.call)
(base_class_clause (type_identifier) @ref.inherit)
`;

const RUBY = `
(method name: (_) @name) @def.method
(singleton_method name: (_) @name) @def.method
(class name: [(constant) @name (scope_resolution name: (_) @name)]) @def.class
(module name: [(constant) @name (scope_resolution name: (_) @name)]) @def.module
(call method: (identifier) @ref.call)
(call receiver: (constant) @ref.call method: (identifier) @_new (#eq? @_new "new"))
(superclass (constant) @ref.inherit)
(superclass (scope_resolution name: (constant) @ref.inherit))
`;

const PHP = `
(namespace_definition name: (namespace_name) @name) @def.module
(interface_declaration name: (name) @name) @def.interface
(trait_declaration name: (name) @name) @def.trait
(class_declaration name: (name) @name) @def.class
(enum_declaration name: (name) @name) @def.enum
(function_definition name: (name) @name) @def.function
(method_declaration name: (name) @name) @def.method
(function_call_expression function: (name) @ref.call)
(function_call_expression function: (qualified_name (name) @ref.call))
(member_call_expression name: (name) @ref.call)
(nullsafe_member_call_expression name: (name) @ref.call)
(scoped_call_expression name: (name) @ref.call)
(object_creation_expression (name) @ref.call)
(object_creation_expression (qualified_name (name) @ref.call))
(base_clause (name) @ref.inherit)
(class_interface_clause (name) @ref.inherit)
`;

const BASH = `
(function_definition name: (word) @name) @def.function
(command name: (command_name (word) @ref.call))
`;

const POWERSHELL = `
(function_statement (function_name) @name) @def.function
(class_statement . (simple_name) @name) @def.class
(class_method_definition (simple_name) @name) @def.method
(command command_name: (command_name) @ref.call)
`;

const KOTLIN = `
(class_declaration name: (identifier) @name) @def.class
(object_declaration name: (identifier) @name) @def.class
(function_declaration name: (identifier) @name) @def.function
(type_alias type: (identifier) @name) @def.type
(call_expression . (identifier) @ref.call)
(call_expression . (navigation_expression (identifier) @ref.call .))
(delegation_specifier (user_type (identifier) @ref.inherit))
(delegation_specifier (constructor_invocation (user_type (identifier) @ref.inherit)))
(user_type (identifier) @ref.type)
`;

const SCALA = `
(class_definition name: (identifier) @name) @def.class
(object_definition name: (identifier) @name) @def.class
(trait_definition name: (identifier) @name) @def.trait
(enum_definition name: (identifier) @name) @def.enum
(function_definition name: (identifier) @name) @def.function
(function_declaration name: (identifier) @name) @def.function
(type_definition name: (type_identifier) @name) @def.type
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression field: (identifier) @ref.call))
(instance_expression (type_identifier) @ref.call)
(extends_clause (type_identifier) @ref.inherit)
(extends_clause (generic_type (type_identifier) @ref.inherit))
`;

const LUA = `
(function_declaration name: (identifier) @name) @def.function
(function_declaration name: (dot_index_expression table: (_) @owner field: (identifier) @name)) @def.function
(function_declaration name: (method_index_expression table: (_) @owner method: (identifier) @name)) @def.method
(assignment_statement (variable_list . name: [(identifier) @name (dot_index_expression field: (identifier) @name)]) (expression_list . value: (function_definition))) @def.function
(function_call name: [(identifier) @ref.call (dot_index_expression field: (identifier) @ref.call) (method_index_expression method: (identifier) @ref.call)])
`;

const ZIG = `
(function_declaration name: (identifier) @name) @def.function
(variable_declaration (identifier) @name [(struct_declaration) (union_declaration) (opaque_declaration)]) @def.struct
(variable_declaration (identifier) @name (enum_declaration)) @def.enum
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression member: (identifier) @ref.call))
`;

const ELIXIR_DEFINERS = `"def" "defp" "defdelegate" "defguard" "defguardp" "defmacro" "defmacrop"`;

const ELIXIR = `
(call target: (identifier) @_kw (arguments (alias) @name) (#any-of? @_kw "defmodule" "defprotocol")) @def.module
(call target: (identifier) @_kw (arguments [(identifier) @name (call target: (identifier) @name) (binary_operator left: (call target: (identifier) @name) operator: "when")]) (#any-of? @_kw ${ELIXIR_DEFINERS})) @def.function
(call target: (identifier) @ref.call (#not-any-of? @ref.call ${ELIXIR_DEFINERS} "defmodule" "defprotocol" "defimpl" "defstruct" "alias" "import" "require" "use"))
(call target: (dot right: (identifier) @ref.call))
`;

export const LANGUAGES = {
    typescript: { grammar: vscode("typescript"), query: TS },
    tsx: { grammar: vscode("tsx"), query: `${TS}${JSX}` },
    javascript: { grammar: vscode("javascript"), query: JS },
    python: { grammar: vscode("python"), query: PYTHON },
    rust: { grammar: vscode("rust"), query: RUST },
    go: { grammar: vscode("go"), query: GO },
    java: { grammar: vscode("java"), query: JAVA },
    csharp: { grammar: vscode("c-sharp"), query: CSHARP },
    c: { grammar: bundled("c"), query: C },
    cpp: { grammar: vscode("cpp"), query: CPP },
    ruby: { grammar: vscode("ruby"), query: RUBY },
    php: { grammar: vscode("php"), query: PHP },
    bash: { grammar: vscode("bash"), query: BASH },
    powershell: { grammar: vscode("powershell"), query: POWERSHELL },
    kotlin: { grammar: bundled("kotlin"), query: KOTLIN },
    scala: { grammar: bundled("scala"), query: SCALA },
    lua: { grammar: bundled("lua"), query: LUA },
    zig: { grammar: bundled("zig"), query: ZIG },
    elixir: { grammar: bundled("elixir"), query: ELIXIR },
} as const satisfies Record<string, { grammar: string; query: string }>;

export type LanguageId = keyof typeof LANGUAGES;

const EXTENSIONS = new Map<string, LanguageId>([
    [".ts", "typescript"], [".mts", "typescript"], [".cts", "typescript"],
    [".tsx", "tsx"],
    [".js", "javascript"], [".mjs", "javascript"], [".cjs", "javascript"], [".jsx", "javascript"],
    [".py", "python"], [".pyi", "python"],
    [".rs", "rust"],
    [".go", "go"],
    [".java", "java"],
    [".cs", "csharp"],
    [".c", "c"],
    [".cc", "cpp"], [".cpp", "cpp"], [".cxx", "cpp"], [".c++", "cpp"], [".h", "cpp"], [".hh", "cpp"], [".hpp", "cpp"], [".hxx", "cpp"], [".ino", "cpp"],
    [".rb", "ruby"], [".rake", "ruby"],
    [".php", "php"],
    [".sh", "bash"], [".bash", "bash"],
    [".ps1", "powershell"], [".psm1", "powershell"],
    [".kt", "kotlin"], [".kts", "kotlin"],
    [".scala", "scala"], [".sc", "scala"],
    [".lua", "lua"],
    [".zig", "zig"],
    [".ex", "elixir"], [".exs", "elixir"],
]);

export function languageForPath(path: string): LanguageId | undefined {
    const dot = path.lastIndexOf(".");
    if (dot <= path.lastIndexOf("/") || path.endsWith(".min.js")) return undefined;
    return EXTENSIONS.get(path.slice(dot).toLowerCase());
}

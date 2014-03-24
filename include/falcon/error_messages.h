/*
   FALCON - The Falcon Programming Language.
   FILE: error_messages.h

   String table used for the engine and the core module messages.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 13 Jan 2009 21:35:17 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004-2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#undef FAL_ERRORDECL

#ifdef FLC_DECLARE_ERROR_TABLE
   #define FAL_ERRORDECL( errid, code, x )         const int errid = code;
#else

   #ifdef FLC_MAKE_ERROR_MESSAGE_SELECTOR
      #define FAL_ERRORDECL( errid, code, str )       case code: return str;
   #else
      #error "Cannot include <falcon/error_messages.h> directly. Please include <falcon/error.h>"
   #endif

#endif

//================================================================
// Error messages.
//

FAL_ERRORDECL( e_none, 0, "No error" );
FAL_ERRORDECL( e_syntax, 1, "Generic syntax error" );
FAL_ERRORDECL( e_unpack_size, 2, "Incompatible unpack size for list assignment" );
FAL_ERRORDECL( e_break_out, 3, "Break outside loops" );
FAL_ERRORDECL( e_continue_out, 4, "Continue outside loops" );
FAL_ERRORDECL( e_div_by_zero, 5, "Division by zero" );
FAL_ERRORDECL( e_mod_by_zero, 6, "Module by zero" );
FAL_ERRORDECL( e_invalid_op, 7, "Invalid operator" );
FAL_ERRORDECL( e_assign_const, 8, "Assignment to a constant" );
FAL_ERRORDECL( e_assign_sym, 9, "Assignment to a non assignable expression" );
FAL_ERRORDECL( e_static_access, 10, "Non-static property accessed statically" );
FAL_ERRORDECL( e_global_notin_func, 11, "Global statement not inside a function" );
FAL_ERRORDECL( e_already_def, 12, "Symbol already defined" );
FAL_ERRORDECL( e_inv_token, 13, "Unrecognized token" );
FAL_ERRORDECL( e_undef_sym, 14, "Undefined symbol" );
FAL_ERRORDECL( e_export_all, 15, "Already exported all" );
FAL_ERRORDECL( e_static_notin_func, 16, "Static related instruction outside a function" );
FAL_ERRORDECL( e_invop, 17, "Unsupported operator for this type" );
FAL_ERRORDECL( e_op_params, 18, "Incompatible operand types" );
FAL_ERRORDECL( e_prop_adef, 19, "Property already defined" );
FAL_ERRORDECL( e_stackuf, 20, "Stack underflow" );
FAL_ERRORDECL( e_stackof, 21, "Stack overflow" );
FAL_ERRORDECL( e_arracc, 22, "Access array out of bounds" );
FAL_ERRORDECL( e_nostartsym, 23, "No startup symbol found" );
FAL_ERRORDECL( e_uncaught, 24, "Explicitly raised item is uncaught" );
FAL_ERRORDECL( e_binload, 25, "System error in loading a binary module" );
FAL_ERRORDECL( e_binstartup, 26, "Binary module has not the 'falcon_module_init' startup procedure" );
FAL_ERRORDECL( e_bininit, 27, "Module cannot be initialized" );
FAL_ERRORDECL( e_modver, 28, "Unrecognized module version" );
FAL_ERRORDECL( e_modformat, 29, "Generic falcon module format error" );
FAL_ERRORDECL( e_modio, 30, "I/O error while loading a module" );
FAL_ERRORDECL( e_unclosed_cs, 31, "Unclosed control structure" );
FAL_ERRORDECL( e_runaway_eof, 32, "Parse error at end of file" );

FAL_ERRORDECL( e_prop_acc, 33, "Requested property not found in object" );
FAL_ERRORDECL( e_deadlock, 34, "Deadlock detected" );
FAL_ERRORDECL( e_prov_name, 35, "Operator ''provides'' must be followed by a symbol name" );
FAL_ERRORDECL( e_init_given, 36, "Constructor already declared" );

FAL_ERRORDECL( e_static_const, 37, "Static member initializers must be a constant expression" );
FAL_ERRORDECL( e_inv_inherit, 38, "Class inherits from a symbol that is not a class" );
FAL_ERRORDECL( e_nonsym_ref, 39, "Trying to get a reference from a constant value" );

FAL_ERRORDECL( e_no_cls_inst, 40, "No internal class found for standalone object" );
FAL_ERRORDECL( e_switch_clash, 41, "Duplicate or clashing switch/select case" );
FAL_ERRORDECL( e_switch_default, 42, "Default block already defined in switch/select" );
FAL_ERRORDECL( e_service_adef, 43, "Service already published" );
FAL_ERRORDECL( e_service_undef, 44, "Required service has not been published" );
FAL_ERRORDECL( e_file_output, 46, "Can't create output file" );
FAL_ERRORDECL( e_domain, 47, "Mathematical domain error" );
FAL_ERRORDECL( e_charRange, 48, "Invalid character while parsing source" );
FAL_ERRORDECL( e_par_close_unbal, 49, "Closing a parenthesis, but never opened" );
FAL_ERRORDECL( e_square_close_unbal, 50, "Closing square bracket, but never opened" );
FAL_ERRORDECL( e_graph_close_unbal, 51, "Closing a bracket, but never opened" );

FAL_ERRORDECL( e_inv_num_format, 52, "Invalid numeric format" );
FAL_ERRORDECL( e_inv_esc_sequence, 53, "Invalid string escape sequence" );
FAL_ERRORDECL( e_numparse_long, 54, "String too long for numeric conversion" );
FAL_ERRORDECL( e_bitwise_op, 55, "Bitwise operation on non-numeric parameters" );
FAL_ERRORDECL( e_switch_body, 56, "Invalid statement in switch body" );
FAL_ERRORDECL( e_select_body, 57, "Invalid statement in select body" );
FAL_ERRORDECL( e_lone_end, 58, "'end' statement without open contexts" );
FAL_ERRORDECL( e_inv_inherit2, 59, "Non-class symbol previously used as class inheritance" );
FAL_ERRORDECL( e_byte_access, 60, "Byte accessor [*x] is read-only" );
FAL_ERRORDECL( e_global_again, 61, "Variable was already global" );

FAL_ERRORDECL( e_malformed_uri, 62, "Malformed or invalid URI" );
FAL_ERRORDECL( e_unknown_vfs, 63, "Unknown virtual file system for scheme part in URI" );
FAL_ERRORDECL( e_modname_inv, 64, "Invalid module logical name" );
FAL_ERRORDECL( e_final_inherit, 65, "Inheriting from a final class" );
FAL_ERRORDECL( e_numparse, 66, "Invalid source data while converting to number" );

FAL_ERRORDECL( e_default_decl, 100, "Syntax error in 'default' statement" );
FAL_ERRORDECL( e_case_decl, 101, "Syntax error in case statement" );
FAL_ERRORDECL( e_switch_decl, 103, "Syntax error in 'switch' statement" );
FAL_ERRORDECL( e_select_decl, 104, "Syntax error in 'select' statement" );
FAL_ERRORDECL( e_case_outside, 105, "Statement 'case' is valid only within switch or select statements" );
FAL_ERRORDECL( e_syn_load, 106, "Syntax error in 'load' directive" );
FAL_ERRORDECL( e_toplevel_func, 107, "Non-anonymous functions must be declared at top level" );
FAL_ERRORDECL( e_toplevel_obj, 108, "Objects must be declared at top level" );
FAL_ERRORDECL( e_toplevel_class, 109, "Classes must be declared at top level" );
FAL_ERRORDECL( e_toplevel_load, 110, "Load directive must be called at top level" );
FAL_ERRORDECL( e_syn_while , 111, "Syntax error in 'while' statement" );
FAL_ERRORDECL( e_syn_if, 112, "Syntax error in 'if' statement" );
FAL_ERRORDECL( e_syn_else, 113, "Syntax error in 'else' statement" );
FAL_ERRORDECL( e_syn_elif, 114, "Syntax error in 'elif' statement" );
FAL_ERRORDECL( e_syn_break, 115, "Syntax error in 'break' statement" );
FAL_ERRORDECL( e_syn_continue, 116, "Syntax error in 'continue' statement" );
FAL_ERRORDECL( e_syn_for, 117, "Syntax error in 'for' statement" );
FAL_ERRORDECL( e_syn_forfirst, 118, "Syntax error in 'forfirst' statement" );
FAL_ERRORDECL( e_syn_forlast, 119, "Syntax error in 'forlast' statement" );
FAL_ERRORDECL( e_syn_formiddle, 120, "Syntax error in 'formiddle' statement" );
FAL_ERRORDECL( e_syn_try, 121, "Syntax error in 'try' statement" );
FAL_ERRORDECL( e_syn_catch, 122, "Syntax error in 'catch' statement" );
FAL_ERRORDECL( e_syn_raise, 123, "Syntax error in 'raise' statement" );
FAL_ERRORDECL( e_syn_funcdecl, 124, "Syntax error in function declaration" );
FAL_ERRORDECL( e_syn_static, 125, "Syntax error in 'static' statement" );
FAL_ERRORDECL( e_syn_launch, 126, "Syntax error in 'launch' statement" );
FAL_ERRORDECL( e_inv_const_val, 127, "Invalid value for constant declaration" );
FAL_ERRORDECL( e_syn_const, 128, "Syntax error in 'const' statement" );
FAL_ERRORDECL( e_syn_export, 129, "Syntax error in 'export' statement" );
FAL_ERRORDECL( e_syn_forin, 130, "Syntax error in 'for..in' statement" );
FAL_ERRORDECL( e_syn_attrdecl, 131, "Syntax error in attribute declaration" );
FAL_ERRORDECL( e_syn_class, 132, "Syntax error in 'class' statement" );
FAL_ERRORDECL( e_syn_object, 133, "Syntax error in 'object' statement" );
FAL_ERRORDECL( e_syn_global, 134, "Syntax error in 'global' statement" );
FAL_ERRORDECL( e_syn_return, 135, "Syntax error in 'return' statement" );
FAL_ERRORDECL( e_syn_arraccess, 136, "Syntax error in array access" );
FAL_ERRORDECL( e_syn_funcall, 137, "Syntax error in function call" );
FAL_ERRORDECL( e_syn_lambda, 138, "Syntax error in 'lambda' statement" );
FAL_ERRORDECL( e_syn_iif, 139, "Syntax error in '?:' expression" );
FAL_ERRORDECL( e_syn_dictdecl, 140, "Syntax error in dictionary declaration" );
FAL_ERRORDECL( e_syn_arraydecl, 141, "Syntax error in array declaration" );
FAL_ERRORDECL( e_syn_def, 142, "Syntax error in 'def' statement" );
FAL_ERRORDECL( e_syn_fordot, 143, "Syntax error in 'for.' statement" );
FAL_ERRORDECL( e_syn_self_print, 144, "Syntax error in fast print statement" );
FAL_ERRORDECL( e_syn_directive, 145, "Syntax error in directive" );
FAL_ERRORDECL( e_syn_import, 146, "Syntax error in import statement" );
FAL_ERRORDECL( e_syn_macro, 147, "Syntax error in macro definition" );
FAL_ERRORDECL( e_syn_macro_call, 148, "Syntax error in macro call" );
FAL_ERRORDECL( e_syn_loop, 149, "Syntax error in loop statement" );
FAL_ERRORDECL( e_syn_end, 150, "Misplaced 'end' keyword" );
FAL_ERRORDECL( e_compile, 151, "Dynamic compilation failed -- details in suberrors" );
FAL_ERRORDECL( e_syn_unpack, 152, "Unpack-array assignment declaration error" );
FAL_ERRORDECL( e_syn_expr, 153, "Error in expression syntax" );
FAL_ERRORDECL( e_syn_finally, 154, "Syntax error in 'finally' statement" );
FAL_ERRORDECL( e_catch_outside, 155, "Statement 'catch' not in a 'try'" );
FAL_ERRORDECL( e_finally_outside, 156, "Statement 'finally' not in a 'try'" );
FAL_ERRORDECL( e_finally_adef, 157, "Statement 'finally' already declared in 'try'" );
FAL_ERRORDECL( e_syn_rangedecl, 158, "Syntax error in range declaration" );
FAL_ERRORDECL( e_syn_evalret_out, 159, "Eval-return (^? or ^=) out of literal context." );

FAL_ERRORDECL( e_not_iterable, 160, "Iterating on non-iterable item." );
FAL_ERRORDECL( e_cmp_unprep, 161, "Compiler not prepared (still needs to be fed with a module)" );
FAL_ERRORDECL( e_not_implemented, 162, "Feature not implemented/not available on this instance" );
FAL_ERRORDECL( e_nl_in_lit, 163, "New line in literal string" );
FAL_ERRORDECL( e_fself_outside, 164, "'fself' outside functions or blocks" );
FAL_ERRORDECL( e_undef_param, 165, "Required parameter not found" );
FAL_ERRORDECL( e_noeffect, 166, "Statement has no effect (at least in part)" );
FAL_ERRORDECL( e_ns_clash, 167, "Can't refer directly to a namespace once declared." );
FAL_ERRORDECL( e_directive_unk, 168, "Unknown directive" );
FAL_ERRORDECL( e_directive_value, 169, "Invalid value for directive" );
FAL_ERRORDECL( e_sm_adef, 170, "State member already defined" );
FAL_ERRORDECL( e_state_adef, 171, "State already defined" );
FAL_ERRORDECL( e_undef_state, 172, "Undefined state" );
FAL_ERRORDECL( e_circular_inh, 173, "Circular inheritance detected" );
FAL_ERRORDECL( e_invop_unb, 174, "Unbound value used in arbitrary operation" );
FAL_ERRORDECL( e_determ_decl, 175, "Determinism status already specified" );
FAL_ERRORDECL( e_dict_key, 176, "The given item cannot be used as a key for a dictionary" );
FAL_ERRORDECL( e_dict_acc, 177, "Key not found in dictionary" );
FAL_ERRORDECL( e_unquote_out, 178, "Unquote out of literal context" );
FAL_ERRORDECL( e_syn_list_decl, 179, "Syntax error in expression list" );

FAL_ERRORDECL( e_catch_clash, 180, "Duplicate type identifier in catch selector" );
FAL_ERRORDECL( e_catch_adef, 181, "Default catch block already defined" );
FAL_ERRORDECL( e_already_forfirst, 182, "Block 'forfirst' already declared" );
FAL_ERRORDECL( e_already_forlast, 183, "Block 'forlast' already declared" );
FAL_ERRORDECL( e_already_formiddle, 184, "Block 'formiddle' already declared" );
FAL_ERRORDECL( e_fordot_outside, 185, "Statement '.=' must be inside a for/in loop" );
FAL_ERRORDECL( e_par_unbal, 186, "Unbalanced parenthesis at end of file" );
FAL_ERRORDECL( e_square_unbal, 187, "Unbalanced square parenthesis at end of file" );
FAL_ERRORDECL( e_unclosed_string, 188, "Unclosed string at end of file" );
FAL_ERRORDECL( e_graph_unbal, 189, "Unbalanced bracket parenthesis at end of file" );
FAL_ERRORDECL( e_syn_cut, 190, "Rule cut '!' not under rule" );
FAL_ERRORDECL( e_forfirst_outside, 191, "Block 'forfirst' declared outside 'for'" );
FAL_ERRORDECL( e_forlast_outside, 192, "Block 'forlast' declared outside 'for'" );
FAL_ERRORDECL( e_formiddle_outside, 193, "Block 'formiddle' declared outside 'for'" );
FAL_ERRORDECL( e_syn_doubt, 194, "Rule doubt '?' not under rule" );
FAL_ERRORDECL( e_catch_invtype, 195, "Invalid type in catch clause" );
FAL_ERRORDECL( e_select_invtype, 196, "Invalid type in select clause" );
FAL_ERRORDECL( e_syn_or, 197, "Rule alternative 'or' not under rule" );
FAL_ERRORDECL( e_for_not_numeric, 198, "For/in loop start, end or step are not numeric values");
FAL_ERRORDECL( e_non_script_error, 199, "Trying to scriptize a non-scriptable error.");


FAL_ERRORDECL( e_io_error, 200, "Generic I/O Error" );
FAL_ERRORDECL( e_io_open, 201, "I/O error: Can't open required resource" );
FAL_ERRORDECL( e_io_creat, 202, "I/O error: Can't create required resource" );
FAL_ERRORDECL( e_io_close, 203, "I/O error: failure during close request" );
FAL_ERRORDECL( e_io_read, 204, "I/O error during read" );
FAL_ERRORDECL( e_io_write, 205, "I/O error during write" );
FAL_ERRORDECL( e_io_seek, 206, "I/O error during seek" );
FAL_ERRORDECL( e_io_ravail, 207, "I/O error during read availability check" );
FAL_ERRORDECL( e_io_wavail, 208, "I/O error during write availability check" );
FAL_ERRORDECL( e_io_invalid, 209, "Stream has been invalidated" );
FAL_ERRORDECL( e_io_unsup, 210, "Unsupported I/O operation on this stream" );
FAL_ERRORDECL( e_io_dup, 211, "Cannot duplicate the source file descriptor" );

FAL_ERRORDECL( e_deser, 220, "Deserialization failed" );
FAL_ERRORDECL( e_deser_eof, 221, "Hit EOF while deserializing" );
FAL_ERRORDECL( e_read_eof, 222, "Reading at eof" );

FAL_ERRORDECL( e_link_error, 250, "Generic link error");
FAL_ERRORDECL( e_loaderror, 251, "Error in loading a module" );
FAL_ERRORDECL( e_invformat, 252, "Invalid or damaged Falcon VM file" );
FAL_ERRORDECL( e_loader_unsupported, 253, "Operation not supported by the module loader" );
FAL_ERRORDECL( e_unrec_file_type, 254, "Unrecognized file type" );
FAL_ERRORDECL( e_nofile, 255, "File not found" );
FAL_ERRORDECL( e_load_already, 256, "Resource already required for load" );
FAL_ERRORDECL( e_import_already, 257, "Imported symbol already declared" );
FAL_ERRORDECL( e_export_already, 258, "Symbol already declared for export" );
FAL_ERRORDECL( e_export_private, 259, "Cannot export a symbol starting with '_' (private)" );
FAL_ERRORDECL( e_syn_import_as, 260, "Import/from 'as' clause supports exactly one single symbol name" );
FAL_ERRORDECL( e_import_already_mod, 261, "Module already generically imported" );
FAL_ERRORDECL( e_mod_notfound, 262, "Module required in load/import not found" );
FAL_ERRORDECL( e_syn_import_spec, 263, "Error in imported symbol specification list" );
FAL_ERRORDECL( e_syn_namespace, 264, "Error in namespace declaration" );
FAL_ERRORDECL( e_syn_namespace_star, 265, "Namespace declarations cannot be extended with '*'" );
FAL_ERRORDECL( e_syn_import_name_star, 266, "Imported symbols cannot contain a star marker (if not at end)" );
FAL_ERRORDECL( e_mod_not_fam, 267, "Fam module has invalid FAM signature" );
FAL_ERRORDECL( e_mod_unsupported_fam, 268, "Unsupported FAM version" );
FAL_ERRORDECL( e_directive_not_allowed, 269, "Directive not allowed in dynamic compilation" );
FAL_ERRORDECL( e_no_main, 270, "Main module has no main function");
FAL_ERRORDECL( e_mod_load_self, 271, "Loader module is loading itself");

FAL_ERRORDECL( e_unknown_encoding, 300, "Unknown encoding name" );
FAL_ERRORDECL( e_enc_fail, 301, "Encoding failed or data not encodable" );
FAL_ERRORDECL( e_dec_fail, 302, "Decode failed or input data not in required format" );
FAL_ERRORDECL( e_membuf_def, 303, "Invalid character in membuf definition" );
FAL_ERRORDECL( e_regex_def, 304, "Error in regular expression definition" );

FAL_ERRORDECL( e_attrib_already, 350, "Attribute already declared" );
FAL_ERRORDECL( e_marshall_not_found, 351, "Message event handler not found" );
FAL_ERRORDECL( e_not_responding, 352, "Summoned entity isn't responding to a mandatory summon");
FAL_ERRORDECL( e_non_delegable, 353, "The target item doesn't support delegation");


FAL_ERRORDECL( e_state, 400, "Unknown state in state-oriented operation" );
FAL_ERRORDECL( e_underflow, 401, "Generic underflow in code flow" );
FAL_ERRORDECL( e_setup, 402, "Required prerequisite steps were not performed" );

FAL_ERRORDECL( e_fmt_convert, 500, "Format not applicable to object" );
FAL_ERRORDECL( e_async_seq_modify, 501, "Underlying object modified while performing a sequential operation" );
FAL_ERRORDECL( e_priv_access, 502, "Access to private member not through 'self'" );
FAL_ERRORDECL( e_noninst_cls, 503, "Target class cannot be instantiated" );
FAL_ERRORDECL( e_unserializable, 504, "Object cannot be serialized (because of inner native data)" );
FAL_ERRORDECL( e_uncloneable, 506, "Uncloneable object, as part of it is not available to VM" );
FAL_ERRORDECL( e_prop_ro, 507, "Tried to write a read-only property" );
FAL_ERRORDECL( e_wait_in_atomic, 508, "VM received a suspension request in an atomic operation" );
FAL_ERRORDECL( e_table_aconf, 509, "Table already configured" );
FAL_ERRORDECL( e_non_callable, 510, "Non callable symbol called" );
FAL_ERRORDECL( e_prop_invalid, 511, "Invalid or inconsistent property value" );
FAL_ERRORDECL( e_invalid_iter, 512, "Invaild iterator status" );
FAL_ERRORDECL( e_iter_outrange, 513, "Iterator out of range" );
FAL_ERRORDECL( e_non_dict_seq, 514, "Given sequence is not a dictionary sequence");
FAL_ERRORDECL( e_miss_iface, 515, "Missing interface: needed method not found");
FAL_ERRORDECL( e_acc_forbidden, 516, "Access forbidden");
FAL_ERRORDECL( e_prop_wo, 517, "Tried to read a write-only property" );
FAL_ERRORDECL( e_prop_loop, 518, "Property accessed inside its accessor" );
FAL_ERRORDECL( e_table_empty, 519, "Operation on an empty table" );
FAL_ERRORDECL( e_tabcol_acc, 520, "Table column not found" );
FAL_ERRORDECL( e_cont_atomic, 521, "Continuation while in atomic mode" );
FAL_ERRORDECL( e_cont_out, 522, "Continuation invoked when already complete" );
FAL_ERRORDECL( e_parse_format, 523, "Input data is not in expected format" );
FAL_ERRORDECL( e_call_loop, 524, "Calling a sequence having itself as callable element" );
FAL_ERRORDECL( e_not_a_class, 525, "Imported symbol is not a class" );
FAL_ERRORDECL( e_abstract_init, 526, "Trying to instance an abstract class" );

FAL_ERRORDECL( e_ctx_ownership, 600, "This object cannot be accessed from this context" );
FAL_ERRORDECL( e_concurrence, 601, "Unauthorized concurrent access to object" );
FAL_ERRORDECL( e_start_thread, 602, "Failed to create a system-level thread" );

FAL_ERRORDECL( e_inv_params, 900, "Invalid parameters" );
FAL_ERRORDECL( e_missing_params, 901, "Mandatory parameter missing" );
FAL_ERRORDECL( e_param_type, 902, "Invalid parameters type" );
FAL_ERRORDECL( e_param_range, 903, "Parameters content invalid/out of range" );
FAL_ERRORDECL( e_param_indir_code, 904, "Parse error in indirect code" );
FAL_ERRORDECL( e_param_strexp_code, 905, "Parse error in expanded string" );
FAL_ERRORDECL( e_param_fmt_code, 906, "Parse error in format specifier" );
FAL_ERRORDECL( e_inv_prop_value, 907, "Invalid value assigned to a property" );
FAL_ERRORDECL( e_param_arity, 908, "Unexpected number of parameters in evaluation" );
FAL_ERRORDECL( e_meta_not_proto, 909, "_meta is not a prototype" );
FAL_ERRORDECL( e_expr_assign, 910, "Given tree step cannot be assigned to the host step" );
FAL_ERRORDECL( e_param_compo, 911, "Positional parameter given after named parameter" );
FAL_ERRORDECL( e_param_noname, 912, "Target item doesn't support named parameters" );
FAL_ERRORDECL( e_param_notfound, 913, "Can't find parameter given by name" );
FAL_ERRORDECL( e_object_not_found, 914, "Accessed entity not found in enumeration or set");
FAL_ERRORDECL( e_base_prop_access, 915, "Invalid property access on a base class");

FAL_ERRORDECL( e_internal, 998, "Intenral error" )
FAL_ERRORDECL( e_paranoid, 999, "Paranoid check failed" );

/* end of error_messages.h */

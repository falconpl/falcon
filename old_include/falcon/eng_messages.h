/*
   FALCON - The Falcon Programming Language.
   FILE: eng_messages.h

   String table used for the engine and the core module messages.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 13 Jan 2009 21:35:17 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004-2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   String table used for the engine and the core module messages.
*/

#undef FAL_ENGMSG
#undef FAL_ERRORDECL

#ifdef FLC_DECLARE_ERROR_TABLE
   // when creating the error enumeration
   #define FAL_ENGMSG( x, y )
   #define FAL_ERRORDECL( errid, code, x )         const int errid = code;
#else

   #ifdef FLC_DECLARE_ENGINE_MSG
      // when globally declaring the stings
      #define FAL_ENGMSG( msg, x )                 int msg;
      #define FAL_ERRORDECL( errid, code, x )      int msg_##errid;
   #else
      #ifdef FLC_REALIZE_ENGINE_MSG
         // when creating the string ids.
         #undef FAL_ENGMSG
         #undef FAL_ERRORDECL
         #define FAL_ENGMSG( msg, str )                  {String *s = new String( str ); s->exported(true); msg = engineStrings->add( s ); }
         #define FAL_ERRORDECL( errid, code, str )       {String *s = new String( str ); s->exported(true); msg_##errid = engineStrings->add( s ); }

      #else
         #ifdef FLC_MAKE_ERROR_MESSAGE_SELECTOR
            #define FAL_ENGMSG( msg, str )
            #define FAL_ERRORDECL( errid, code, str )       case errid: return Falcon::Engine::getMessage( msg_##errid );
         #else

            // The normal case
            #define FAL_ENGMSG( msg, x )                 extern int msg;
            #define FAL_ERRORDECL( errid, code, x )      extern int msg_##errid;
         #endif
      #endif
   #endif

#endif


//================================================================
// Generic engine messages
//

FAL_ENGMSG( msg_banner, "The Falcon Programming Language" );
FAL_ENGMSG( msg_unknown_error, "Unrecognized error code" );
FAL_ENGMSG( msg_io_curdir, "I/O Error in reading current directory" );

//================================================================
// RTL/ Core module MESSAGES
//
FAL_ENGMSG( rtl_start_outrange, "start position out of range" );
FAL_ENGMSG( rtl_array_missing, "required an array, a start and an end position" );
FAL_ENGMSG( rtl_inv_startend, "invalid start/end positions" );
FAL_ENGMSG( rtl_cmdp_0, "parameter array contains non string elements" );
FAL_ENGMSG( rtl_emptyarr, "parameter array is empty" );
FAL_ENGMSG( rtl_iterator_not_found, "\"Iterator\" class not found in VM" );
FAL_ENGMSG( rtl_invalid_iter, "Given item is not a valid iterator for the collection" );
FAL_ENGMSG( rtl_marshall_not_cb, "Marshalled event name must be a string as first element in the given array" );
FAL_ENGMSG( rtl_invalid_path, "Invalid path" );
FAL_ENGMSG( rtl_invalid_uri, "Invalid URI" );
FAL_ENGMSG( rtl_no_tabhead, "First row of the table must be a header" );
FAL_ENGMSG( rtl_invalid_tabhead, "Table header must be composed of strings or future bindings" );
FAL_ENGMSG( rtl_invalid_order, "Table order must be greater than zero" );
FAL_ENGMSG( rtl_invalid_tabrow, "Row inserted in table has different order" );
FAL_ENGMSG( rtl_broken_table, "The table changed during a const operation" );
FAL_ENGMSG( rtl_uncallable_col, "Given column contains some uncallable items" );
FAL_ENGMSG( rtl_no_page, "Page ID not present in table" );
FAL_ENGMSG( rtl_tabhead_given, "Table heading already given" );
FAL_ENGMSG( rtl_string_empty, "Refill string cannot be empty" );
FAL_ENGMSG( rtl_row_out_of_bounds, "Row larger than current page size" );
FAL_ENGMSG( rtl_buffer_full, "Given memory buffer is full" );
FAL_ENGMSG( rtl_zero_size, "Given size less or equal to zero" );

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
FAL_ERRORDECL( e_assign_sym, 9, "Assignment to a non assignable symbol" );
FAL_ERRORDECL( e_static_call, 10, "Non-static method called statically" );
FAL_ERRORDECL( e_global_notin_func, 11, "Global statement not inside a function" );
FAL_ERRORDECL( e_already_def, 12, "Symbol already defined" );
FAL_ERRORDECL( e_inv_token, 13, "Unrecognized token" );
FAL_ERRORDECL( e_undef_sym, 14, "Undefined symbol" );
FAL_ERRORDECL( e_export_all, 15, "Already exported all" );
FAL_ERRORDECL( e_static_notin_func, 16, "Static related instruction outside a function" );
FAL_ERRORDECL( e_invop, 17, "Invalid operands given opcode" );
FAL_ERRORDECL( e_prop_pinit, 18, "Property definition after init definition" );
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
FAL_ERRORDECL( e_nonsym_ref, 39, "Trying to get a reference from something that's not a symbol" );

FAL_ERRORDECL( e_no_cls_inst, 40, "No internal class found for standalone object" );
FAL_ERRORDECL( e_switch_clash, 41, "Duplicate or clashing switch case" );
FAL_ERRORDECL( e_switch_default, 42, "Default block already defined in switch" );
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
FAL_ERRORDECL( e_inv_inherit2, 59, "Inheritance from more than one subtree of reflected classes" );
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

FAL_ERRORDECL( e_catch_clash, 150, "Duplicate type identifier in catch selector" );
FAL_ERRORDECL( e_catch_adef, 151, "Default catch block already defined" );
FAL_ERRORDECL( e_already_forfirst, 152, "Block 'forfirst' already declared" );
FAL_ERRORDECL( e_already_forlast, 153, "Block 'forlast' already declared" );
FAL_ERRORDECL( e_already_formiddle, 154, "Block 'formiddle' already declared" );
FAL_ERRORDECL( e_fordot_outside, 155, "Statement '.=' must be inside a for/in loop" );
FAL_ERRORDECL( e_par_unbal, 156, "Unbalanced parenthesis at end of file" );
FAL_ERRORDECL( e_square_unbal, 157, "Unbalanced square parenthesis at end of file" );
FAL_ERRORDECL( e_unclosed_string, 158, "Unclosed string at end of file" );
FAL_ERRORDECL( e_graph_unbal, 159, "Unbalanced bracket parenthesis at end of file" );

FAL_ERRORDECL( e_cmp_unprep, 161, "Compiler not prepared (still needs to be fed with a module)" );
FAL_ERRORDECL( e_not_implemented, 162, "Feature not implemented/not available on this instance" );
FAL_ERRORDECL( e_nl_in_lit, 163, "New line in literal string" );
FAL_ERRORDECL( e_fself_outside, 164, "'fself' outside functions or blocks" );
FAL_ERRORDECL( e_undef_param, 165, "Required parameter not found" );
FAL_ERRORDECL( e_noeffect, 166, "Statement has no effect (at least in part)" );
FAL_ERRORDECL( e_ns_clash, 167, "Clash in namespaces aliasing" );
FAL_ERRORDECL( e_directive_unk, 168, "Unknown directive" );
FAL_ERRORDECL( e_directive_value, 169, "Invalid value for directive" );
FAL_ERRORDECL( e_sm_adef, 170, "State member already defined" );
FAL_ERRORDECL( e_state_adef, 171, "State already defined" );
FAL_ERRORDECL( e_undef_state, 172, "Undefined state" );
FAL_ERRORDECL( e_circular_inh, 173, "Circular inheritance detected" );
FAL_ERRORDECL( e_invop_unb, 174, "Unbound value used in arbitrary operation" );


FAL_ERRORDECL( e_open_file, 200, "Can't open file" );
FAL_ERRORDECL( e_loaderror, 201, "Error in loading a module" );
FAL_ERRORDECL( e_nofile, 202, "File not found" );
FAL_ERRORDECL( e_invformat, 203, "Invalid or damaged Falcon VM file" );
FAL_ERRORDECL( e_loader_unsupported, 204, "Operation not supported by the module loader" );
FAL_ERRORDECL( e_io_error, 205, "Generic I/O Error" );
FAL_ERRORDECL( e_unknown_encoding, 206, "Unknown encoding name" );
FAL_ERRORDECL( e_unrec_file_type, 207, "Unrecognized file type" );
FAL_ERRORDECL( e_io_unsup, 208, "Unrecognized file type" );
FAL_ERRORDECL( e_io_invalid, 209, "Unrecognized file type" );
FAL_ERRORDECL( e_deser_eof, 210, "Hit EOF while deserializing" );
FAL_ERRORDECL( e_search_eof, 211, "Search operation failed or item not found" );

FAL_ERRORDECL( e_fmt_convert, 500, "Format not applicable to object" );
FAL_ERRORDECL( e_interrupted, 501, "Asynchronous wait interruption" );
FAL_ERRORDECL( e_priv_access, 502, "Access to private member not through 'self'" );
FAL_ERRORDECL( e_noninst_cls, 503, "Target class cannot be instantiated" );
FAL_ERRORDECL( e_unserializable, 504, "Object cannot be serialized (because of inner native data)" );
FAL_ERRORDECL( e_uncloneable, 506, "Uncloneable object, as part of it is not available to VM" );
FAL_ERRORDECL( e_prop_ro, 507, "Tried to write a read-only property" );
FAL_ERRORDECL( e_wait_in_atomic, 508, "VM received a suspension request in an atomic operation" );
FAL_ERRORDECL( e_table_aconf, 509, "Table already configured" );
FAL_ERRORDECL( e_non_callable, 510, "Non callable symbol called" );
FAL_ERRORDECL( e_prop_invalid, 511, "Invalid or inconsistent property value" );
FAL_ERRORDECL( e_invalid_iter, 512, "Invalid iterator applied to sequence method" );
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

FAL_ERRORDECL( e_inv_params, 900, "Invalid parameters" );
FAL_ERRORDECL( e_missing_params, 901, "Mandatory parameter missing" );
FAL_ERRORDECL( e_param_type, 902, "Invalid parameters type" );
FAL_ERRORDECL( e_param_range, 903, "Parameters content invalid/out of range" );
FAL_ERRORDECL( e_param_indir_code, 904, "Parse error in indirect code" );
FAL_ERRORDECL( e_param_strexp_code, 905, "Parse error in expanded string" );
FAL_ERRORDECL( e_param_fmt_code, 906, "Parse error in format specifier" );


/* end of eng_messages.h */

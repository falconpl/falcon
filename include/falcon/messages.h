/*
   FALCON - The Falcon Programming Language.
   FILE: messages.h
   $Id: messages.h,v 1.15 2007/08/03 13:17:05 jonnymind Exp $

   Standard engine and basic module messages
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar feb 13 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Standard engine and basic module messages.
   This file contains both the definition of the messages and the
   reference to the standard built-in message tables for english
   and other built-in languages.
*/

#ifndef flc_messages_H
#define flc_messages_H

namespace Falcon {
namespace msg {
   enum messages {
      hello = 0,
      err_syntax,
      err_unpack_size,
      err_break_out,
      err_continue_out,
      err_div_by_zero,
      err_mod_by_zero,
      err_invalid_op,
      err_assign_const,
      err_assign_sym,
      //10
      err_repeated,
      err_global_notin_func,
      err_already_def,
      err_non_callable,
      err_invalid_cmp,
      err_export_undef,
      err_export_all,
      err_misplaced_stat,
      err_enter_outside,
      err_leave_outside,
      //20
      err_static_notin_func,
      err_self_outclass,
      err_sender_outclass,
      err_undef_sym,
      err_invops,
      err_no_local,
      err_too_entry,
      err_end_no_loc,
      err_import_out,
      err_no_import,
      //30
      err_too_locals,
      err_too_params,
      err_switch_again,
      err_switch_case,
      err_switch_end,
      err_inv_setstate,
      err_prop_no_class,
      err_prop_pinit,
      err_prop_adef,
      err_too_props,
      // 40
      err_from_adef,
      err_too_froms,
      err_invopcode,
      err_invop,
      err_stackuf,
      err_stackof,
      err_arracc,
      err_nostartsym,
      err_uncaught,
      err_binload,
      err_binstartup,
      err_bininit,
      err_modver,
      err_modformat,
      err_modio,
      err_unclosed_cs,
      err_runaway_eof,
      err_cmp_unprep,
      err_undef_label,
      err_prop_acc,
      err_deadlock,
      err_prov_name,
      err_dup_case,
      err_init_given,
      err_static_const,
      err_str_noid,
      err_inv_inherit,
      err_nonsym_ref,
      err_state_adef,
      err_invalid_sjmp,
      err_no_attrib,
      err_no_cls_inst,
      err_pass_outside,
      err_switch_clash,
      err_switch_default,
      err_for_user_error,
      err_global_again,
      err_service_adef,
      err_service_undef,
      err_uncloneable,
      err_param_outside,
      err_file_output,
      err_domain,
      err_charRange,
      err_par_close_unbal,
      err_square_close_unbal,
      err_inv_num_format,
      err_inv_esc_sequence,
      err_eol_string,
      err_inv_token,
      err_inv_direct,
      err_byte_access,
      err_numparse_long,
      err_numparse,
      err_no_class,
      err_bitwise_op,
      err_case_decl,
      err_switch_body,
      err_select_body,
      err_default_decl,
      err_lone_end,
      err_switch_decl,
      err_select_decl,
      err_case_outside,
      err_syn_load,
      err_toplevel_func,
      err_toplevel_obj,
      err_toplevel_class,
      err_toplevel_load,
      err_syn_while ,
      err_syn_if,
      err_syn_else,
      err_syn_elif,
      err_syn_break,
      err_syn_continue,
      err_syn_for,
      err_syn_forfirst,
      err_syn_forlast,
      err_syn_formiddle,
      err_syn_try,
      err_syn_catch,
      err_syn_raise,
      err_syn_funcdecl,
      err_syn_static,
      err_syn_state,
      err_syn_launch,
      err_syn_pass,
      err_inv_const_val,
      err_syn_const,
      err_syn_export,
      err_syn_attributes,
      err_enter_notavar,
      err_syn_enter,
      err_syn_leave,
      err_syn_forin,
      err_syn_pass_in,
      err_leave_notanexp,
      err_inv_attrib,
      err_syn_class,
      err_syn_hasdef,
      err_syn_object,
      err_syn_global,
      err_syn_return,
      err_syn_arraccess,
      err_syn_funcall,
      err_syn_lambda,
      err_syn_iif,
      err_syn_dictdecl,
      err_syn_arraydecl,
      err_syn_give,
      err_syn_def,
      err_syn_fordot,
      err_syn_self_print,
      err_syn_directive,
      err_syn_import,

      err_nl_in_lit,
      err_catch_clash,
      err_catch_adef,
      err_fmt_convert,
      err_already_forfirst,
      err_already_forlast,
      err_already_formiddle,
      err_fordot_outside,
      err_interrupted,
      err_priv_access,
      err_par_unbal,
      err_square_unbal,
      err_unclosed_string,
      err_directive_unk,
      err_directive_value,

      err_open_file,
      err_loaderror,
      err_nofile,
      err_invformat,
      err_loader_unsupported,


      err_inv_params,
      err_missing_params,
      err_param_type,
      err_param_range,
      err_param_indir_code,
      err_param_strexp_code,
      err_param_fmt_code,

      /* Generic messages */
      unrecognized_err,

      core_001,
      core_002
   };
}



}

#endif

/* end of messages.h */

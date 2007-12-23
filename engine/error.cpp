/*
   FALCON - The Falcon Programming Language
   FILE: error.cpp
   $Id: error.cpp,v 1.21 2007/08/18 17:59:33 jonnymind Exp $

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 28 2006
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
   Error management.
*/

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/messages.h>
#include <falcon/engstrings.h>
#include <falcon/sys.h>
#include <falcon/vm.h>
#include <falcon/cobject.h>

namespace Falcon {

const String &errorDesc( int code )
{
   switch( code )
   {
      case e_none: return getMessage( msg::unrecognized_err );
      case e_syntax: return getMessage( msg::err_syntax );
      case e_unpack_size: return getMessage( msg::err_unpack_size );
      case e_break_out: return getMessage( msg::err_break_out );
      case e_continue_out: return getMessage( msg::err_continue_out);
      case e_div_by_zero: return getMessage( msg::err_div_by_zero);
      case e_mod_by_zero: return getMessage( msg::err_mod_by_zero);
      case e_invalid_op: return getMessage( msg::err_invalid_op);
      case e_assign_const: return getMessage( msg::err_assign_const);
      case e_assign_sym: return getMessage( msg::err_assign_sym);
      case e_repeated: return getMessage( msg::err_repeated);
      case e_global_notin_func: return getMessage( msg::err_global_notin_func);
      case e_already_def: return getMessage( msg::err_already_def);
      case e_non_callable: return getMessage( msg::err_non_callable);
      case e_invalid_cmp: return getMessage( msg::err_invalid_cmp);
      case e_export_undef: return getMessage( msg::err_export_undef);
      case e_export_all: return getMessage( msg::err_export_all);
      case e_misplaced_stat: return getMessage( msg::err_misplaced_stat);
      case e_enter_outside: return getMessage( msg::err_enter_outside);
      case e_leave_outside: return getMessage( msg::err_leave_outside);
      case e_static_notin_func: return getMessage( msg::err_static_notin_func);
      case e_self_outclass: return getMessage( msg::err_self_outclass);
      case e_sender_outclass: return getMessage( msg::err_sender_outclass);
      case e_undef_sym: return getMessage( msg::err_undef_sym);
      case e_invops: return getMessage( msg::err_invops);
      case e_no_local: return getMessage( msg::err_no_local);
      case e_too_entry: return getMessage( msg::err_too_entry);
      case e_end_no_loc: return getMessage( msg::err_end_no_loc);
      case e_import_out: return getMessage( msg::err_import_out);
      case e_no_import: return getMessage( msg::err_no_import);
      case e_too_locals: return getMessage( msg::err_too_locals);
      case e_too_params: return getMessage( msg::err_too_params);
      case e_switch_again: return getMessage( msg::err_switch_again);
      case e_switch_case: return getMessage( msg::err_switch_case);
      case e_switch_end: return getMessage( msg::err_switch_end);
      case e_inv_setstate: return getMessage( msg::err_inv_setstate);
      case e_prop_no_class: return getMessage( msg::err_prop_no_class);
      case e_prop_pinit: return getMessage( msg::err_prop_pinit);
      case e_prop_adef: return getMessage( msg::err_prop_adef);
      case e_too_props: return getMessage( msg::err_too_props);
      case e_from_adef: return getMessage( msg::err_from_adef);
      case e_too_froms: return getMessage( msg::err_too_froms);
      case e_invopcode: return getMessage( msg::err_invopcode);
      case e_invop: return getMessage( msg::err_invop);
      case e_stackuf: return getMessage( msg::err_stackuf);
      case e_stackof: return getMessage( msg::err_stackof);
      case e_arracc: return getMessage( msg::err_arracc);
      case e_nostartsym: return getMessage( msg::err_nostartsym);
      case e_uncaught: return getMessage( msg::err_uncaught);
      case e_binload: return getMessage( msg::err_binload);
      case e_binstartup: return getMessage( msg::err_binstartup);
      case e_bininit: return getMessage( msg::err_bininit);
      case e_modver: return getMessage( msg::err_modver);
      case e_modformat: return getMessage( msg::err_modformat);
      case e_modio: return getMessage( msg::err_modio);
      case e_unclosed_cs: return getMessage( msg::err_unclosed_cs);
      case e_runaway_eof: return getMessage( msg::err_runaway_eof);
      case e_attrib_space: return getMessage( msg::err_attrib_space);
      case e_undef_label: return getMessage( msg::err_undef_label);
      case e_prop_acc: return getMessage( msg::err_prop_acc);
      case e_deadlock: return getMessage( msg::err_deadlock);
      case e_prov_name: return getMessage( msg::err_prov_name);
      case e_dup_case: return getMessage( msg::err_dup_case);
      case e_init_given: return getMessage( msg::err_init_given);
      case e_static_const: return getMessage( msg::err_static_const);
      case e_str_noid: return getMessage( msg::err_str_noid);
      case e_inv_inherit: return getMessage( msg::err_inv_inherit);
      case e_nonsym_ref: return getMessage( msg::err_nonsym_ref);
      case e_state_adef: return getMessage( msg::err_state_adef);
      case e_invalid_sjmp: return getMessage( msg::err_invalid_sjmp);
      case e_no_attrib: return getMessage( msg::err_no_attrib);
      case e_no_cls_inst: return getMessage( msg::err_no_cls_inst);
      case e_pass_outside: return getMessage( msg::err_pass_outside);
      case e_switch_clash: return getMessage( msg::err_switch_clash);
      case e_switch_default: return getMessage( msg::err_switch_default);
      case e_for_user_error: return getMessage( msg::err_for_user_error);
      case e_global_again: return getMessage( msg::err_global_again);
      case e_service_adef: return getMessage( msg::err_service_adef);
      case e_service_undef: return getMessage( msg::err_service_undef);
      case e_uncloneable: return getMessage( msg::err_uncloneable);
      case e_param_outside: return getMessage( msg::err_param_outside);
      case e_file_output: return getMessage( msg::err_file_output);
      case e_domain: return getMessage( msg::err_domain);
      case e_charRange: return getMessage( msg::err_charRange);
      case e_par_close_unbal: return getMessage( msg::err_par_close_unbal);
      case e_square_close_unbal: return getMessage( msg::err_square_close_unbal);
      case e_inv_num_format: return getMessage( msg::err_inv_num_format);
      case e_inv_esc_sequence: return getMessage( msg::err_inv_esc_sequence);
      case e_eol_string: return getMessage( msg::err_eol_string);
      case e_inv_token: return getMessage( msg::err_inv_token);
      case e_inv_direct: return getMessage( msg::err_inv_direct);
      case e_byte_access: return getMessage( msg::err_byte_access );
      case e_numparse_long: return getMessage( msg::err_numparse_long );
      case e_numparse: return getMessage( msg::err_numparse );
      case e_no_class: return getMessage( msg::err_no_class );
      case e_bitwise_op: return getMessage( msg::err_bitwise_op );
      case e_case_decl: return getMessage( msg::err_case_decl );
      case e_switch_body: return getMessage( msg::err_switch_body );
      case e_select_body: return getMessage( msg::err_select_body );
      case e_default_decl: return getMessage( msg::err_default_decl );
      case e_lone_end: return getMessage( msg::err_lone_end );
      case e_switch_decl: return getMessage( msg::err_switch_decl );
      case e_select_decl: return getMessage( msg::err_select_decl );
      case e_case_outside: return getMessage( msg::err_case_outside );
      case e_syn_load: return getMessage( msg::err_syn_load );
      case e_toplevel_func: return getMessage( msg::err_toplevel_func );
      case e_toplevel_obj: return getMessage( msg::err_toplevel_obj );
      case e_toplevel_class: return getMessage( msg::err_toplevel_class );
      case e_toplevel_load: return getMessage( msg::err_toplevel_load );
      case e_syn_while : return getMessage( msg::err_syn_while  );
      case e_syn_if: return getMessage( msg::err_syn_if );
      case e_syn_else: return getMessage( msg::err_syn_else );
      case e_syn_elif: return getMessage( msg::err_syn_elif );
      case e_syn_break: return getMessage( msg::err_syn_break );
      case e_syn_continue: return getMessage( msg::err_syn_continue );
      case e_syn_for: return getMessage( msg::err_syn_for );
      case e_syn_forfirst: return getMessage( msg::err_syn_forfirst );
      case e_syn_forlast: return getMessage( msg::err_syn_forlast );
      case e_syn_forall: return getMessage( msg::err_syn_forall );
      case e_syn_give: return getMessage( msg::err_syn_give );
      case e_syn_try: return getMessage( msg::err_syn_try );
      case e_syn_catch: return getMessage( msg::err_syn_catch );
      case e_syn_raise: return getMessage( msg::err_syn_raise );
      case e_syn_funcdecl: return getMessage( msg::err_syn_funcdecl );
      case e_syn_static: return getMessage( msg::err_syn_static );
      case e_syn_state: return getMessage( msg::err_syn_state );
      case e_syn_launch: return getMessage( msg::err_syn_launch );
      case e_syn_pass: return getMessage( msg::err_syn_pass );
      case e_inv_const_val: return getMessage( msg::err_inv_const_val );
      case e_syn_const: return getMessage( msg::err_syn_const );
      case e_syn_export: return getMessage( msg::err_syn_export );
      case e_syn_attributes: return getMessage( msg::err_syn_attributes );
      case e_enter_notavar: return getMessage( msg::err_enter_notavar );
      case e_syn_enter: return getMessage( msg::err_syn_enter );
      case e_syn_leave: return getMessage( msg::err_syn_leave );
      case e_syn_forin: return getMessage( msg::err_syn_forin );
      case e_syn_pass_in: return getMessage( msg::err_syn_pass_in );
      case e_leave_notanexp: return getMessage( msg::err_leave_notanexp );
      case e_inv_attrib: return getMessage( msg::err_inv_attrib );
      case e_syn_class: return getMessage( msg::err_syn_class );
      case e_syn_hasdef: return getMessage( msg::err_syn_hasdef );
      case e_syn_object: return getMessage( msg::err_syn_object );
      case e_syn_global: return getMessage( msg::err_syn_global );
      case e_syn_return: return getMessage( msg::err_syn_return );
      case e_syn_arraccess: return getMessage( msg::err_syn_arraccess );
      case e_syn_funcall: return getMessage( msg::err_syn_funcall );
      case e_syn_lambda: return getMessage( msg::err_syn_lambda );
      case e_syn_iif: return getMessage( msg::err_syn_iif );
      case e_syn_dictdecl: return getMessage( msg::err_syn_dictdecl );
      case e_syn_arraydecl: return getMessage( msg::err_syn_arraydecl );
      case e_syn_fordot: return getMessage( msg::err_syn_fordot );
      case e_syn_self_print: return getMessage( msg::err_syn_self_print );
      case e_par_unbal: return getMessage( msg::err_par_unbal );
      case e_square_unbal: return getMessage( msg::err_square_unbal );
      case e_unclosed_string: return getMessage( msg::err_unclosed_string );

      case e_nl_in_lit: return getMessage( msg::err_nl_in_lit );
      case e_catch_clash: return getMessage( msg::err_catch_clash );
      case e_catch_adef: return getMessage( msg::err_catch_adef );
      case e_syn_def: return getMessage( msg::err_syn_def );
      case e_fmt_convert: return getMessage( msg::err_fmt_convert );
      case e_fordot_outside: return getMessage( msg::err_fordot_outside );
      case e_interrupted: return getMessage( msg::err_interrupted );
      case e_priv_access: return getMessage( msg::err_priv_access );

      case e_already_forfirst: return getMessage( msg::err_already_forfirst );
      case e_already_forlast: return getMessage( msg::err_already_forlast );
      case e_already_forall: return getMessage( msg::err_already_forall );


      case e_open_file: return getMessage( msg::err_open_file);
      case e_loaderror: return getMessage( msg::err_loaderror);
      case e_nofile: return getMessage( msg::err_nofile);
      case e_invformat: return getMessage( msg::err_invformat );
      case e_loader_unsupported: return getMessage( msg::err_loader_unsupported );


      case e_inv_params: return getMessage( msg::err_inv_params );
      case e_missing_params: return getMessage( msg::err_missing_params );
      case e_param_type: return getMessage( msg::err_param_type );
      case e_param_range: return getMessage( msg::err_param_range );
      case e_param_indir_code: return getMessage( msg::err_param_indir_code );
      case e_param_strexp_code: return getMessage( msg::err_param_strexp_code );
      case e_param_fmt_code: return getMessage( msg::err_param_fmt_code );

      default:
         return getMessage( msg::unrecognized_err );
   }
}


String &TraceStep::toString( String &target ) const
{
   target += m_module + "." + m_symbol + ":";
   target.writeNumber( (int64) m_line );
   target += "(PC:";
   switch( m_pc )
   {
      case VMachine::i_pc_call_external: target += "ext"; break;
      case VMachine::i_pc_call_external_return: target += "ext.r"; break;
      case VMachine::i_pc_redo_request: target += "redo"; break;
      case VMachine::i_pc_call_external_ctor: target += "ext.c"; break;
      case VMachine::i_pc_call_external_ctor_return: target += "ext.cr"; break;
      default:
         target.writeNumber( (int64) m_pc );
   }

   target += ")";

	return target;
}

//==================================================
// Error
//==================================================

Error::Error( const Error &e ):
   m_nextError( 0 ),
   m_LastNextError( 0 )
{
   m_errorCode = e.m_errorCode;
   m_line = e.m_line;
   m_pc = e.m_pc;
   m_character = e.m_character;
   m_sysError = e.m_sysError;
   m_origin = e.m_origin;

   m_catchable = e.m_catchable;

   m_description = e.m_description;
   m_extra = e.m_extra;
   m_module = e.m_module;
   m_symbol = e.m_symbol;
   m_raised = e.m_raised;
   m_className = e.m_className;


   m_refCount = 1;

   ListElement *step_i = m_steps.begin();
   while( step_i != 0 )
   {
      TraceStep *step = (TraceStep *) step_i->data();
      addTrace( step->module(), step->symbol(), step->line(), step->pcounter() );
      step_i = step_i->next();
   }
}


Error::~Error()
{
   ListElement *step_i = m_steps.begin();
   while( step_i != 0 )
   {
      TraceStep *step = (TraceStep *) step_i->data();
      delete step;
      step_i = step_i->next();
   }

   Error *ptr = m_nextError;
   while( ptr != 0 )
   {
      Error *ptrnext = ptr->m_nextError;
      ptr->m_nextError = 0;
      ptr->decref();

      ptr = ptrnext;
   }
}

void Error::incref()
{
   m_refCount++;
}

void Error::decref()
{
   --m_refCount;
	if( m_refCount == 0 )
   {
		delete this;
   }
}

String &Error::toString( String &target ) const
{
   heading( target );
   target += "\n";

   if ( ! m_steps.empty() )
   {
      target += "  Traceback:\n";

      ListElement *iter = m_steps.begin();
      while( iter != 0 )
      {
          target += "   ";
          TraceStep *step = (TraceStep *) iter->data();
          step->toString( target );
          target += "\n";
          iter = iter->next();
      }
   }

   // recursive stringation
   if ( m_nextError != 0 )
      m_nextError->toString( target );

   return target;
}


String &Error::heading( String &target ) const
{
   target += m_className;
   target += " ";

   switch( m_origin )
   {
   case e_orig_compiler: target += "CO"; break;
   case e_orig_assembler: target += "AS"; break;
   case e_orig_loader: target += "LD"; break;
   case e_orig_vm: target += "VM"; break;
   case e_orig_runtime: target += "RT"; break;
   case e_orig_mod: target += "MD"; break;
   case e_orig_script: target += "SS"; break;
   }

   uint32 ecode = (uint32) m_errorCode;

   for ( int number = 1000; number > 0; number /= 10 )
   {
      int64 cipher = ecode / number;
      ecode %= number;
      target.writeNumber( cipher );
   }

   if ( m_sysError != 0 )
   {
      target += "(sys: ";
      target.writeNumber( (int64) m_sysError );
      String temp;
      Sys::_describeError( m_sysError, temp );
      target += " " + temp;
      target += ")";
   }

   if( m_line != 0 || m_module.size() != 0 )
      target += " at ";

   if ( m_module.size() != 0 )
   {
      target += m_module;
      if ( m_symbol.size() != 0 )
         target += "." + m_symbol;
      target += ":";
   }

   if ( m_line != 0 )
      target.writeNumber( (int64) m_line );

   if ( m_character != 0 ) {
      target += "/";
      target.writeNumber( (int64) m_character );
   }

   if ( m_pc != 0 )
   {
      target += "(PC:";
       switch( m_pc )
      {
         case VMachine::i_pc_call_external: target += "ext"; break;
         case VMachine::i_pc_call_external_return: target += "ext.r"; break;
         case VMachine::i_pc_redo_request: target += "redo"; break;
         case VMachine::i_pc_call_external_ctor: target += "ext.c"; break;
         case VMachine::i_pc_call_external_ctor_return: target += "ext.cr"; break;
         default:
            target.writeNumber( (int64) m_pc );
      }
      target += ")";
   }

   if ( m_description.size() > 0 )
   {
      target += ": " + m_description;
   }
   else {
      target += ": " + errorDesc( m_errorCode );
   }

   if ( m_extra.size() > 0 )
   {
      target += " (" + m_extra + ")";
   }

   if ( ! m_raised.isNil() )
   {
      String temp;
      m_raised.toString( temp );
      target += "\n"+ temp;
   }

   return target;
}


void Error::addTrace( const String &module, const String &symbol, uint32 line, uint32 pc )
{
   m_steps.pushBack( new TraceStep( module, symbol, line, pc ) );
   m_stepIter = m_steps.begin();
}

void Error::appendSubError( Error *error )
{
   if ( m_LastNextError == 0 )
   {
      m_LastNextError = m_nextError = error;
   }
   else {
      m_LastNextError->m_nextError = error;
      m_LastNextError = error;
   }

   error->incref();
}

bool Error::nextStep( String &module, String &symbol, uint32 &line, uint32 &pc )
{
   if ( m_steps.empty() || m_stepIter == 0 )
      return false;

   TraceStep *step = (TraceStep *) m_stepIter->data();
   module = step->module();
   symbol = step->symbol();
   line = step->line();
   pc = step->pcounter();

   m_stepIter = m_stepIter->next();

   return true;
}

void Error::rewindStep()
{
   if ( ! m_steps.empty() )
      m_stepIter = m_steps.begin();
}

CoreObject *Error::scriptize( VMachine *vm )
{
   Item *error_class = vm->findGlobalItem( m_className );

   if ( error_class == 0 || ! error_class->isClass() ) {
      Error *error = new GenericError(
         ErrorParam( e_undef_sym, __LINE__).origin(e_orig_vm).module( "core.Error" ).
         symbol( "Error::scriptize" ).extra( m_className ).hard()
         );
      vm->raiseRTError( error );
      return 0;
   }

   ErrorCarrier *carrier = new ErrorCarrier( this );
   CoreObject *cobject = error_class->asClass()->createInstance();
   cobject->setUserData( carrier );

   // scriptize sub-errors
   if (m_nextError != 0)
   {
      CoreArray *errorList = new CoreArray( vm );

      Error *ptr = m_nextError;
      while( ptr != 0 )
      {
         ErrorCarrier *subcarrier = new ErrorCarrier( ptr );

         CoreObject *subobject = error_class->asClass()->createInstance();
         subobject->setUserData( subcarrier );

         errorList->append( subobject );
         ptr = ptr->m_nextError;
      }

      cobject->setProperty( "subErrors", errorList );
   }
   return cobject;
}




ErrorCarrier::ErrorCarrier( Error *carried ):
   m_error( carried )
{
   carried->incref();
   switch( m_error->origin() )
   {
      case e_orig_compiler: m_origin = "compiler"; break;
      case e_orig_assembler: m_origin = "assembler"; break;
      case e_orig_loader: m_origin = "loader"; break;
      case e_orig_vm: m_origin = "vm"; break;
      case e_orig_script: m_origin = "script"; break;
      case e_orig_runtime: m_origin = "runtime"; break;
      case e_orig_mod: m_origin = "module"; break;
   }
}


ErrorCarrier::~ErrorCarrier()
{
   m_error->decref();
}

bool ErrorCarrier::isReflective()
{
   return true;
}

void ErrorCarrier::getProperty( const String &propName, Item &prop )
{
   if ( m_error == 0 )
      return;

   if ( propName == "code" )
      prop = (int64) m_error->errorCode();
   else if ( propName == "description" )
      prop = const_cast<String *>(&m_error->errorDescription());
   else if ( propName == "message" )
      prop = const_cast<String *>( &m_error->extraDescription() );
   else if ( propName == "systemError" )
      prop = (int64) m_error->systemError();
   else if ( propName == "origin" )
      prop = const_cast<String *>( &m_origin );
   else if ( propName == "module" )
      prop = const_cast<String *>( &m_error->module() );
   else if ( propName == "symbol" )
      prop = const_cast<String *>( &m_error->symbol() );
   else if ( propName == "line" )
      prop = (int64) m_error->line();
   else if ( propName == "pc" )
      prop = (int64) m_error->pcounter();
   else if ( propName == "fsError" )
      prop = (int64) m_error->systemError();

}

void ErrorCarrier::setProperty( const String &propName, Item &prop )
{
   if ( m_error == 0 )
      return;

   if ( propName == "code" )
      m_error->errorCode( (int) prop.forceInteger() );
   else if ( propName == "description" && prop.isString() )
      m_error->errorDescription( *prop.asString() );
   else if ( propName == "message" && prop.isString() )
      m_error->extraDescription( *prop.asString() );
   else if ( propName == "origin" && prop.isString() )
   {
      String &origin = *prop.asString();
      if( origin == "compiler" )
      {
         m_origin = origin;
         m_error->origin( e_orig_compiler );
      }
      else if( origin == "assembler" )
      {
         m_origin = origin;
         m_error->origin( e_orig_assembler );
      }
      else if( origin == "loader" )
      {
         m_origin = origin;
         m_error->origin( e_orig_loader );
      }
      else if( origin == "vm" )
      {
         m_origin = origin;
         m_error->origin( e_orig_vm );
      }
      else if( origin == "script" )
      {
         m_origin = origin;
         m_error->origin( e_orig_script );
      }
      else if( origin == "runtime" )
      {
         m_origin = origin;
         m_error->origin( e_orig_runtime );
      }
      else if( origin == "module" )
      {
         m_origin = origin;
         m_error->origin( e_orig_mod);
      }
   }
   else if ( propName == "module" && prop.isString() )
      m_error->module( *prop.asString() );
   else if ( propName == "symbol" && prop.isString() )
      m_error->symbol( *prop.asString() );
   else if ( propName == "line" )
      m_error->line( (uint32) prop.forceInteger() );
   else if ( propName == "pc" )
      m_error->pcounter( (uint32) prop.forceInteger() );
   else if ( propName == "systemError" )
      m_error->systemError( (uint32) prop.forceInteger() );
}


}


/* end of error.cpp */

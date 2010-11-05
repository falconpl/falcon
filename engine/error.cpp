/*
   FALCON - The Falcon Programming Language
   FILE: error.cpp

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 28 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Error management.
*/

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/sys.h>
#include <falcon/vm.h>
#include <falcon/coreobject.h>

#include <falcon/eng_messages.h>

namespace Falcon {

const String &errorDesc( int code )
{
   switch( code )
   {
      #define FLC_MAKE_ERROR_MESSAGE_SELECTOR
      #include <falcon/eng_messages.h>
   }

   return Engine::getMessage( msg_unknown_error );
}


String &TraceStep::toString( String &target ) const
{
   if ( m_modpath.size() )
   {
      target += "\"" + m_modpath + "\" "; 
   }

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
   m_LastNextError( 0 ),
   m_boxed( 0 )
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

   if ( e.m_boxed != 0 )
   {
      boxError(m_boxed);
   }

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

   if ( m_boxed != 0 )
   {
      m_boxed->decref();
   }
}

void Error::incref()
{
   atomicInc( m_refCount );
}

void Error::decref()
{
   if( atomicDec( m_refCount ) <= 0 )
   {
      delete this;
   }
}

String &Error::toString( String &target ) const
{
   if ( m_boxed != 0 )
   {
      target += m_boxed->toString();
      target += "  =====================================================\n";
      target += "  Boxed in ";
   }

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
   {
      m_nextError->toString( target );
   }

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
   default: target += "??";
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

void Error::addTrace( const String &module, const String &modpath, const String &symbol, uint32 line, uint32 pc )
{
   m_steps.pushBack( new TraceStep( module, modpath, symbol, line, pc ) );
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


void Error::boxError( Error *error )
{
   if ( m_boxed != 0 )
   {
      m_boxed->decref();
   }

   m_boxed = error;
   if ( error != 0 )
   {
      error->incref();
   }
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
   Item *error_class = vm->findWKI( m_className );
   // in case of 0, try with global items
   if ( error_class == 0 )
      error_class = vm->findGlobalItem( m_className );

   if ( error_class == 0 || ! error_class->isClass() ) {
      throw new GenericError(
         ErrorParam( e_undef_sym, __LINE__)
         .origin(e_orig_vm)
         .module( "core.Error" ).
         symbol( "Error::scriptize" )
         .extra( m_className ).hard()
         );
   }

   // CreateInstance will use ErrorObject, which increfs to us.
   CoreObject *cobject = error_class->asClass()->createInstance( this );
   return cobject;
}

Error *Error::clone() const
{
   return new Error( *this );
}

//==============================================================================
// Reflections

namespace core {

void Error_code_rfrom( CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_FROM( error, errorCode );
}

void Error_description_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_FROM( error, errorDescription );
}

void Error_message_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_FROM( error, extraDescription );
}

void Error_systemError_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_FROM( error, systemError );
}

void Error_origin_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);

   String origin;

   switch( error->origin() )
   {
      case e_orig_compiler: origin = "compiler"; break;
      case e_orig_assembler: origin = "assembler"; break;
      case e_orig_loader: origin = "loader"; break;
      case e_orig_vm: origin = "vm"; break;
      case e_orig_script: origin = "script"; break;
      case e_orig_runtime: origin = "runtime"; break;
      case e_orig_mod: origin = "module"; break;
      default: origin = "unknown";
   }

   property = new CoreString( origin );
}

void Error_module_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_FROM( error, module );
}

void Error_symbol_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_FROM( error, symbol );
}

void Error_line_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_FROM( error, line );
}

void Error_pc_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_FROM( error, pcounter );
}

void Error_boxed_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   if ( error->getBoxedError() != 0 )
   {
      VMachine* vm = VMachine::getCurrent();
      fassert( vm != 0 );
      property =  error->getBoxedError()->scriptize(vm);
   }
   else
   {
      property.setNil();
   }
}

void Error_subErrors_rfrom(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   VMachine* vm = VMachine::getCurrent();
   fassert( vm != 0 );

   // scriptize sub-errors
   Error *ptr = error->subError();
   if ( ptr != 0)
   {
      CoreArray *errorList = new CoreArray();

      do
      {
         // CreateInstance will use ErrorObject, which increfs ptr.
         CoreObject *subobject = ptr->scriptize( vm );

         errorList->append( subobject );
         ptr = ptr->subError();
      }
      while( ptr != 0 );

      property = errorList;
   }
   else
      property.setNil();
}


void Error_code_rto( CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_TO( error, errorCode );
}

void Error_description_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_TO( error, errorDescription );
}

void Error_message_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_TO( error, extraDescription );
}

void Error_systemError_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_TO( error, systemError );
}

void Error_origin_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);

   if ( property.isString() )
   {
      String &origin = *property.asString();
      if( origin == "compiler" )
      {
         error->origin( e_orig_compiler );
      }
      else if( origin == "assembler" )
      {
         error->origin( e_orig_assembler );
      }
      else if( origin == "loader" )
      {
         error->origin( e_orig_loader );
      }
      else if( origin == "vm" )
      {
         error->origin( e_orig_vm );
      }
      else if( origin == "script" )
      {
         error->origin( e_orig_script );
      }
      else if( origin == "runtime" )
      {
         error->origin( e_orig_runtime );
      }
      else if( origin == "module" )
      {
         error->origin( e_orig_mod);
      }
      else {
         throw new ParamError( ErrorParam( e_param_range ) );
      }

      return;
   }

   throw new ParamError( ErrorParam( e_inv_params ).extra( "S" ) );
}

void Error_module_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_TO( error, module );
}

void Error_symbol_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_STRING_TO( error, symbol );
}

void Error_line_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_TO( error, line );
}

void Error_pc_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   FALCON_REFLECT_INTEGER_TO( error, pcounter );
}

void Error_boxed_rto(CoreObject *instance, void *userData, Item &property, const PropEntry& )
{
   Error *error = static_cast<Error *>(userData);
   if ( property.isObject() && property.asObject()->derivedFrom("Error") )
   {
      error->boxError( static_cast<Error*>(property.asObject()->getUserData()) );
   }
}

//============================================================
// Reflector

ErrorObject::ErrorObject( const CoreClass* cls, Error *err ):
   CRObject( cls )
{
   if ( err != 0 )
   {
      err->incref();
      setUserData( err );
   }
}

ErrorObject::~ErrorObject()
{
   if ( m_user_data != 0 )
      getError()->decref();
}

void ErrorObject::gcMark( uint32 mark )
{
   Error* error = getError();
   if ( error != 0 )
      memPool->markItem( const_cast<Item&>(error->raised()) );
}


ErrorObject *ErrorObject::clone() const
{
   return new ErrorObject( m_generatedBy, getError() );
}

//========================================================
// Factory function
//

CoreObject* ErrorObjectFactory( const CoreClass *cls, void *user_data, bool )
{
      return new ErrorObject( cls, (Error *) user_data );
}

}} // namespace Falcon::core 

/* end of error.cpp */

/*
   FALCON - The Falcon Programming Language
   FILE: error.cpp

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 21:15:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/tracestep.h>
#include <falcon/mt.h>
#include <falcon/sys.h>
#include <falcon/class.h>

#include <falcon/engine.h>

#include <falcon/error_messages.h>

namespace Falcon {

static const String errorDesc( int code )
{
   static String unk("Unknown error");

   switch( code )
   {
      #define FLC_MAKE_ERROR_MESSAGE_SELECTOR
      #include <falcon/error_messages.h>
   }

   return unk;
}

//==================================================
// Error
//==================================================

Error::Error( Class* handler, const ErrorParam &params ):
   m_refCount( 1 ),
   m_errorCode( params.m_errorCode ),
   m_description( params.m_description ),
   m_extra( params.m_extra ),
   m_symbol( params.m_symbol ),
   m_module( params.m_module ),
   m_line( params.m_line ),
   m_sysError( params.m_sysError ),
   m_origin( params.m_origin ),
   m_catchable( params.m_catchable ),
   m_handler( handler )
{}

Error::~Error()
{
   std::deque<Error*>::const_iterator iter = m_subErrors.begin();
   while( iter != m_subErrors.end() )
   {
      (*iter)->decref();
      ++iter;
   }
}

void Error::incref() const
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

void Error::describe( String &target ) const
{
   heading( target );
   target += "\n";

   if ( ! m_steps.empty() )
   {
      target += "  Traceback:\n";

      std::deque<TraceStep>::const_iterator iter = m_steps.begin();
      while( iter != m_steps.end() )
      {
          target += "   ";
          const TraceStep& step = *iter;
          step.toString( target );
          target += "\n";
          ++iter;
      }
   }

   if (! m_subErrors.empty() )
   {
      target += "  Because of:\n";
      std::deque<Error*>::const_iterator iter = m_subErrors.begin();
      while( iter != m_subErrors.end() )
      {
         (*iter)->describe( target );
         ++iter;
      }
   }
}


String &Error::heading( String &target ) const
{
   target += m_className;
   target += " ";

   switch( m_origin )
   {
   case ErrorParam::e_orig_compiler: target += "CO"; break;
   case ErrorParam::e_orig_assembler: target += "AS"; break;
   case ErrorParam::e_orig_loader: target += "LD"; break;
   case ErrorParam::e_orig_vm: target += "VM"; break;
   case ErrorParam::e_orig_runtime: target += "RT"; break;
   case ErrorParam::e_orig_mod: target += "MD"; break;
   case ErrorParam::e_orig_script: target += "SS"; break;
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
      m_raised.describe( temp );
      target += "\n"+ temp;
   }

   return target;
}

void Error::addTrace( const TraceStep& tb )
{
   m_steps.push_back( tb );
}


void Error::appendSubError( Error *error )
{
   error->incref();
   m_subErrors.push_back( error );
}



void Error::scriptize( Item& tgt )
{
   incref();
   tgt.setDeep( Engine::instance()->collector()->store( m_handler, this ) );
}

void Error::enumerateSteps( Error::StepEnumerator &rator ) const
{
   std::deque<TraceStep>::const_iterator iter = m_steps.begin();
   while( iter != m_steps.end() )
   {
      const TraceStep& ts = *iter;
      ++iter;
      bool last = iter == m_steps.end();
      if( ! rator( ts, last ) ) break;
   }
}

void Error::enumerateErrors( Error::ErrorEnumerator &rator ) const
{
   std::deque<Error*>::const_iterator iter = m_subErrors.begin();
   while( iter != m_subErrors.end() )
   {
      Error* error = *iter;
      ++iter;
      bool last = iter == m_subErrors.end();
      if( ! rator( error, last ) ) break;
   }
}


Error* Error::getBoxedError() const
{
   if( m_subErrors.empty() )
      return 0;
   return m_subErrors.front();
}

/** Return the name of this error class.
 Set in the constructcor.
 */
const String& Error::className() const
{
   return m_handler->name();
}

bool Error::hasTraceback() const
{
   return ! m_steps.empty();
}


} // namespace Falcon

/* end of error.cpp */


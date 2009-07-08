/*
   FALCON - The Falcon Programming Language.
   FILE: int_mode.cpp

   Falcon compiler and interpreter - interactive mode
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "int_mode.h"
#include <falcon/intcomp.h>


IntMode::IntMode( AppFalcon* owner ):
   m_owner( owner )
{}


void IntMode::read_line( Stream *in, String &line, uint32 maxSize )
{
   line.reserve( maxSize );
   line.size(0);
   uint32 chr;
   while ( line.length() < maxSize && in->get( chr ) )
   {
      if ( chr == '\r' )
         continue;
      if ( chr == '\n' )
         break;
      line += chr;
   }
}

void IntMode::run()
{
   ModuleLoader ml;
   m_owner->prepareLoader( ml );

   VMachineWrapper intcomp_vm;
   intcomp_vm->link( core_module_init() );

   InteractiveCompiler comp( &ml, intcomp_vm.vm() );
   comp.setInteractive( true );

   Stream *stdOut = m_owner->m_stdOut;
   Stream *stdIn = m_owner->m_stdIn;

   stdOut->writeString("\nWelcome to Falcon interactive mode.\n" );
   stdOut->writeString("Write statements directly at the prompt; when finished press " );
   #ifdef FALCON_SYSTEM_WIN
      stdOut->writeString("CTRL+Z" );
   #else
      stdOut->writeString("CTRL+D" );
   #endif

   stdOut->writeString(" to exit\n" );

   InteractiveCompiler::t_ret_type lastRet = InteractiveCompiler::e_nothing;
   String line, pline, codeSlice;
   while( stdIn->good() && ! stdIn->eof() )
   {
      const char *prompt = (
            lastRet == InteractiveCompiler::e_more
            ||lastRet == InteractiveCompiler::e_incomplete
            )
         ? "... " : ">>> ";

      stdOut->writeString( prompt );
      stdOut->flush();

      read_line( stdIn, pline, 1024 );
      if ( pline.size() > 0 )
      {
         if( pline.getCharAt( pline.length() -1 ) == '\\' )
         {
            lastRet = InteractiveCompiler::e_more;
            pline.setCharAt( pline.length() - 1, ' ');
            line += pline;
            continue;
         }
         else
            line += pline;

         InteractiveCompiler::t_ret_type lastRet1 = InteractiveCompiler::e_nothing;
         
         bool hadError = false;
         try
         {
            lastRet1 = comp.compileNext( codeSlice + line + "\n" );
         }
         catch( Error *err )
         {
            String temp = err->toString();
            err->decref();
            stdOut->writeString( temp );
            hadError = true;
         }

         switch( lastRet1 )
         {
            case InteractiveCompiler::e_more:
               codeSlice += line + "\n";
               break;

            case InteractiveCompiler::e_incomplete:
               // is it incomplete because of '\\' at end?
               if ( line.getCharAt( line.length()-1 ) == '\\' )
                  line.setCharAt( line.length()-1, ' ' );
               codeSlice += line + "\n";
               break;
           
            case InteractiveCompiler::e_terminated:
               stdOut->writeString( "falcon: Terminated\n\n");
               stdOut->flush();
               return;
               
            case InteractiveCompiler::e_call:
               if ( comp.vm()->regA().isNil() )
               {
                  codeSlice.size(0);
                  break;
               }
               // fallthrough

            case InteractiveCompiler::e_expression:
               {
                  String temp;
                  comp.vm()->itemToString( temp, &comp.vm()->regA() );
                  stdOut->writeString( ": " + temp + "\n" );
               }
               // fallthrough
               
            default:
               codeSlice.size(0);
         }
         
         // do not clear the input on error.
         // -- so the user can try to re-add the last line.
         if ( ! hadError )
         {
            // maintain previous status if having a compilation error.
            lastRet = lastRet1;
         }
         line.size(0);
      }
      // else just continue.
   }

   stdOut->writeString( "\r     \n\n");
   stdOut->flush();
}

/* end of int_mode.cpp */

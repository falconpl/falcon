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
#include <falcon/src_lexer.h>

#include <stdio.h>

using namespace Falcon;

IntMode::IntMode( AppFalcon* owner ):
   m_owner( owner )
{}



void IntMode::run()
{
   ModuleLoader ml;
   m_owner->prepareLoader( ml );

   VMachineWrapper intcomp_vm;
   intcomp_vm->link( core_module_init() );
   Item* describe = intcomp_vm->findGlobalItem("describe");
   fassert( describe != 0 );

   InteractiveCompiler comp( &ml, intcomp_vm.vm() );
   comp.setInteractive( true );

   Stream *stdOut = m_owner->m_stdOut;
   Stream *stdIn = m_owner->m_stdIn;

   stdOut->writeString("\n===NOTICE===\n" );
   stdOut->writeString("Interactive mode is currently UNDER DEVELOPMENT.\n" );

   stdOut->writeString("\nWelcome to Falcon interactive mode.\n" );
   stdOut->writeString("Write statements directly at the prompt; when finished press " );
   #ifdef FALCON_SYSTEM_WIN
      stdOut->writeString("CTRL+Z" );
   #else
      stdOut->writeString("CTRL+D" );
   #endif

   stdOut->writeString(" to exit\n" );
   stdOut->flush();

   InteractiveCompiler::t_ret_type lastRet = InteractiveCompiler::e_nothing;
   String line, pline, codeSlice;
   int linenum = 1;
   while( stdIn->good() && ! stdIn->eof() )
   {
      const char *prompt = (
            lastRet == InteractiveCompiler::e_more
            ||lastRet == InteractiveCompiler::e_incomplete
            )
         ? "... " : ">>> ";

      read_line(pline, prompt);
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
         
         try
         {
            comp.lexer()->line( linenum );
            lastRet1 = comp.compileNext( codeSlice + line + "\n" );
         }
         catch( Error *err )
         {
            String temp = err->toString();
            err->decref();
            stdOut->writeString( temp );
            stdOut->flush();
            // in case of error detected at context end, close it.
            line.trim();
            if( line == "end" || line.endsWith( "]" )
            		  || line.endsWith( "}" ) || line.endsWith( ")" ) )
            {
            	codeSlice.size( 0 );
            	lastRet = InteractiveCompiler::e_nothing;
            	linenum = 1;
            }
            line.size(0);
            continue;
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
                  comp.vm()->pushParameter( comp.vm()->regA() );
                  comp.vm()->callItem( *describe, 1 );
                  stdOut->writeString( ": " + *comp.vm()->regA().asString() + "\n" );
                  stdOut->flush();
               }
               // fallthrough
               
            default:
               codeSlice.size(0);
               linenum = 0;
         }
         

		// maintain previous status if having a compilation error.
		lastRet = lastRet1;
		line.size( 0 );
		linenum++;
      }
      // else just continue.
   }

   stdOut->writeString( "\r     \n\n");
   stdOut->flush();
}

/* end of int_mode.cpp */

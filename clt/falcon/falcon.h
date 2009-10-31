/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.h

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Main Falcon.
*/

#ifndef FALCON_CLT_H
#define FALCON_CLT_H

#include <falcon/engine.h>
using namespace Falcon; // TODO using namespace in a header file, evil!

#include "options.h"
#include "int_mode.h"

/** Typical embedding applications. */
class AppFalcon
{
   FalconOptions m_options;
   int m_exitval;
   int m_errors;
   int m_script_pos;
   int m_argc;
   char** m_argv;

   String getSrcEncoding();
   String getLoadPath();
   String getIoEncoding();
   Module* loadInput( ModuleLoader &ml );

   void applyDirectives ( Compiler &compiler );
   void applyConstants ( Compiler &compiler );
   void readyStreams();

   Stream* openOutputStream( const String &ext );

public:
   /** Prepares the application.
      Sets up global values and starts the falcon engine.
   */
   AppFalcon();

   /** Shuts down the Falcon application.
      This destroys the global data and shuts down the engine.
   */
   ~AppFalcon();

   /** Checks the parameters and determines if the run step must be performed.
      If the program should stop after the setup (for any non-critical error),
      the function sreturns false.

      In case of critical errors, it raises a falcon String,
   */
   bool setup( int argc, char* argv[] );

   /** Perform the operations required by the options. */
   void run();

   void compileTLTable();
   void generateAssembly();
   void generateTree();
   void buildModule();
   void runModule();
   void makeInteractive();
   void prepareLoader( ModuleLoader &ml );


   void terminate();
   int exitval() const { return m_exitval; }
   void exitval( int exitVal ) { m_exitval = exitVal; }

   Stream* m_stdIn;
   Stream* m_stdOut;
   Stream* m_stdErr;

};

#endif

/* end of falcon.h */

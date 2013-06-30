/*
   FALCON - The Falcon Programming Language.
   FILE: debugger.cpp

   Falcon virtual machine -- Basic standard debugger-in-a-box
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#undef SRC
#define SRC "engine/debugger.cpp"

#include <falcon/debugger.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>
#include <falcon/treestep.h>
#include <falcon/function.h>

#include <falcon/stream.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>

#include <falcon/stringstream.h>
#include <falcon/dyncompiler.h>
#include <falcon/syntree.h>

namespace Falcon {


class Debugger::PStepPostEval: public PStep
{
public:
   PStepPostEval() { apply = apply_; }
   virtual ~PStepPostEval() {}
   virtual void describeTo( String& target ) const { target = "PStepPostEval"; }

private:
   static void apply_( const PStep*, VMContext* ctx )
   {
      ctx->popCode();
      TextWriter tw( ctx->vm()->stdErr() );
      tw.write("**: ");
      tw.writeLine( ctx->topData().describe() );
      ctx->popData(2);
   }
};


Debugger::Debugger() :
    m_hello(true)
{
   m_stepPostEval = new PStepPostEval;
}

Debugger::~Debugger()
{
   delete m_stepPostEval;
}

void Debugger::onBreak( Process* p, Processor*, VMContext* ctx )
{
   MESSAGE("Debugger::onBreak -- Debugger invoked.");

   // get the standard input stream.
   Stream* input = p->vm()->stdIn();
   Stream* output = p->vm()->stdErr();

   TextReader tr(input);
   TextWriter wr( output );

   if( m_hello )
   {
      wr.writeLine( "*: Falcon debugger engaged, type \"help\" for help." );
      m_hello = false;
   }

   printCode( wr, ctx );

   bool more;
   do
   {
      MESSAGE1("Debugger::onBreak -- Accepting command");
      wr.write( "*> " );
      wr.flush();

      String line;
      tr.readLine(line, 1024);

      more = parseCommand( wr, line, ctx );
   }
   while( more );

   MESSAGE("Debugger::onBreak -- Exiting.");
}


bool Debugger::parseCommand( TextWriter& wr, const String& line, VMContext* ctx )
{
   const PStep* stepBreak = &Engine::instance()->stdSteps()->m_breakpoint;

   TRACE("Debugger::parseCommand -- parsing '%s'", line.c_ize() );

   // by default, we don't want to loop immediately.
   bool cont = false;
   // whether to save this line or not
   bool save = true;

   if( line == "quit" )
   {
      ctx->process()->terminate();
      save = false;
   }
   else if( line == "exit" )
   {
      MESSAGE("Debugger::parseCommand -- Exiting the debugger on user request" );
      cont = false;
   }
   else if( line == "help" || line == "?" )
   {
      wr.write( "*: Command list:\n"
               "*: help: This help.\n"

               "*: quit: Terminate current Falcon VM process.\n"
               "*: exit: terminate the debugger and resume execution.\n"

               "*: step: Proceed in the next expression (step in).\n"
               "*: next: Execute next expression without descending into it (step over).\n"
               "*: cont: Resume execution up to the next breakpoint.\n"

               "*: eval <expr>: Evaluate given expression -- can change variables.\n"
               "*: src: Locate current code in module and function.\n"

               "*: Entering an empty line repeats the previous command.\n"
               );
      cont = true;
   }
   else if( line == "step" )
   {
      wr.write( "*: single step\n" );
      ctx->setBreakpointEvent();
   }
   else if( line == "next" )
   {
      wr.writeLine( "*: big step" );
      CodeFrame temp = ctx->currentCode();
      ctx->resetCode( stepBreak );
      ctx->pushCode( temp.m_step );
      ctx->currentCode().m_seqId = temp.m_seqId;
      ctx->currentCode().m_dataDepth = temp.m_dataDepth;
      ctx->currentCode().m_dynsDepth = temp.m_dynsDepth;
   }
   else if( line == "cont" )
   {
      wr.writeLine( "*: Continuing." );
   }
   else if( line.startsWith("eval ") )
   {
      Stream* sinput = new StringStream( line.subString(5) );
      TextReader* reader = new TextReader(sinput);
      DynCompiler dynComp( ctx );

      try
      {
         SynTree* st = dynComp.compile( reader );
         ctx->pushCode(stepBreak);
         ctx->pushData( FALCON_GC_HANDLE(st) );
         ctx->pushCode(m_stepPostEval);
         ctx->pushCode(st);
      }
      catch( Error* err )
      {
         wr.write("*!: ");
         wr.writeLine(err->describe(true));
         err->decref();
         cont = true;
      }
      reader->decref();
   }
   else if( line == "src" )
   {
      printLoc(wr, ctx);
      printCode(wr, ctx);
      cont = true;
   }
   else if( line == "" )
   {
      if (m_lastCommand != "")
      {
         cont = Debugger::parseCommand(wr, m_lastCommand, ctx);
      }
      else {
         save = false;
      }
   }
   else {
      wr.writeLine("*!: Unknown command");
      cont = true;
      save = false;
   }

   if( save )
   {
      m_lastCommand = line;
   }

   TRACE1("Debugger::parseCommand -- return with continuation '%s'", (cont? "true": "false") );
   return cont;
}


void Debugger::printCode(TextWriter& wr, VMContext* ctx)
{
   const PStep* step = ctx->nextStep();
   if( step->sr().line() == 0 )
   {
      wr.writeLine(step->describe());
   }
   else {
      const TreeStep* ts = static_cast<const TreeStep*>(step);
      wr.write(String("*: line ").N(step->sr().line()).A(": "));
      ts->render(&wr,1);
   }
}


void Debugger::printLoc(TextWriter& wr, VMContext* ctx)
{
   Function* func = ctx->currentFrame().m_function;
   if( func == 0 )
   {
      wr.writeLine("*: No current function context.");
   }
   else {
      wr.writeLine("*: in " + func->locate());
   }
}

}

/* end of debugger.cpp */

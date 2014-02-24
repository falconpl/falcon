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
#include <falcon/symbol.h>

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

               "*: stack [N]: Display the data in the stack (top N positions, 0 for all).\n"
               "*: dyns [N]: Display the data in the dynamic stack (top N positions, 0 for all).\n"
               "*: code [N]: Display the data in the code stack (top N positions, 0 for all).\n"
               "*: call [N]: Display the data in the call stack (top N positions, 0 for all).\n"

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
      CodeFrame temp;
      temp.m_seqId = ctx->currentCode().m_seqId;
      temp.m_dataDepth = ctx->currentCode().m_dataDepth;
      temp.m_dynsDepth = ctx->currentCode().m_dynsDepth;
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
   else if( line == "stack" )
   {
      displayStack( wr, ctx, 1 );
      cont = true;
   }
   else if( line.startsWith("stack ") )
   {
      int64 depth;
      if( ! line.parseInt(depth,6) || depth < 0 )
      {
         wr.write( "*: invalid depth \n" );
      }
      else {
         displayStack( wr, ctx, depth );
      }
      cont = true;
   }
   else if( line == "dyns" )
   {
      displayDyns( wr, ctx, 1 );
      cont = true;
   }
   else if( line.startsWith("dyns ") )
   {
     int64 depth;
     if( ! line.parseInt(depth,5) || depth < 0 )
     {
        wr.write( "*: invalid depth \n" );
     }
     else {
        displayDyns( wr, ctx, depth );
     }
     cont = true;
   }
   else if( line == "call" )
   {
      displayCall( wr, ctx, 1 );
   }
   else if( line.startsWith("call ") )
   {
     int64 depth;
     if( ! line.parseInt(depth,5) || depth < 0 )
     {
        wr.write( "*: invalid depth \n" );
     }
     else {
        displayCall( wr, ctx, depth );
     }
     cont = true;
   }
   else if( line == "code" )
   {
     displayCode( wr, ctx, 1 );
   }
   else if( line.startsWith("code ") )
   {
     int64 depth;
     if( ! line.parseInt(depth,5) || depth < 0 )
     {
        wr.write( "*: invalid depth \n" );
     }
     else {
        displayCode( wr, ctx, depth );
     }
     cont = true;
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


void Debugger::displayStack( TextWriter& wr, VMContext* ctx, int64 depth )
{
   wr.write( String("*: Data stack size ").N((int64)ctx->dataSize()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->dataSize()) || (depth > 0 && top < depth ) )
   {
      Item* item = &ctx->opcodeParam(top);
      Class* cls = 0;
      void* data = 0;
      item->forceClassInst(cls, data);

      String temp;
      cls->describe(data, temp, 3, 128);

      wr.write( String("*: ").N(top).A(": ").A(temp).A("\n") );
      ++top;
   }
}

void Debugger::displayDyns( TextWriter& wr, VMContext* ctx, int64 depth )
{
   wr.write( String("*: Dyns stack size ").N((int64)ctx->dynsDepth()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->dynsDepth()) || (depth > 0 && top < depth ) )
   {
      VMContext::DynsData* dd = ctx->dynsAt(top);
      Item* item = dd->m_value;
      Class* cls = 0;
      void* data = 0;
      item->forceClassInst(cls, data);

      String temp;
      cls->describe(data, temp, 3, 128);

      wr.write( String("*: ").N(top).A(": ").A(dd->m_sym->name()).A("=(").H((int64)item,true).A(") ").A(temp).A("\n") );
      ++top;
   }
}


void Debugger::displayCode( TextWriter& wr, VMContext* ctx, int64 depth )
{
   wr.write( String("*: Code stack size ").N((int64)ctx->codeDepth()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->codeDepth()) || (depth > 0 && top < depth ) )
   {
      CodeFrame* cf = ctx->codeAt(top);
      String temp;
      cf->m_step->describeTo(temp);
      wr.write( String("*: ").N(top).A(": ").A(temp).A("(").N(cf->m_seqId).A(")\n") );
      ++top;
   }
}

void Debugger::displayCall( TextWriter& wr, VMContext* ctx, int64 depth )
{
   wr.write( String("*: Call stack size ").N((int64)ctx->callDepth()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->callDepth()) || (depth > 0 && top < depth ) )
   {
      CallFrame* cf = &ctx->callerFrame(top);
      wr.write( String("*: ").N(top).A(": ").A(cf->m_function->fullName()).A("(").A(cf->m_function->locate()).A(")\n") );
      ++top;
   }
}


}

/* end of debugger.cpp */

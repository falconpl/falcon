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
#include <falcon/stringtok.h>
#include <falcon/inspector.h>

#include <falcon/stream.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>

#include <falcon/stringstream.h>
#include <falcon/dyncompiler.h>
#include <falcon/syntree.h>
#include <falcon/module.h>
#include <falcon/function.h>
#include <falcon/globalsmap.h>

#include <falcon/psteps/stmttry.h>

#include <vector>

namespace Falcon {

class Debugger::PStepCatcher:public SynTree
{
public:
   PStepCatcher() {apply=apply_;}
   virtual ~PStepCatcher() {}
   virtual void describeTo( String& target ) const { target = "Debugger::PStepCatcher"; }
   virtual PStepCatcher* clone() const { return new PStepCatcher; }
   virtual void render( TextWriter* tw, int32 ) const { tw->write("/*catcher*/\n"); }

private:

   static void apply_( const PStep*, VMContext* ctx )
   {
      ctx->popCode();
      ctx->popData(1); // we pushed the syntree as well to keep it in the GC
      if( ctx->thrownError() != 0)
      {
         TextWriter tw( ctx->vm()->stdErr() );
         tw.write( "**: Evaluation operation caused the following error\n" );
         tw.write( ctx->thrownError()->describe(true) );
         tw.write( "\n" );
         ctx->setBreakpointEvent();
      }
   }
};

class Debugger::PStepPostEval: public StmtTry
{
public:
   PStepPostEval()
   {
      apply = apply_;
      PStepCatcher* catcher = new PStepCatcher;
      catchSelect().append(catcher);
   }

   virtual ~PStepPostEval() {}
   virtual void describeTo( String& target ) const { target = "Debugger::PStepPostEval"; }

private:
   static void apply_( const PStep*, VMContext* ctx )
   {
      ctx->popCode();
      TextWriter tw( ctx->vm()->stdErr() );
      tw.write("**: ");
      tw.writeLine( ctx->topData().describe() );
      ctx->popData(2); // we pushed the syntree as well to keep it in the GC
      ctx->setBreakpointEvent();
   }
};


class Debugger::PStepAfterNext: public StmtTry
{
public:
   PStepAfterNext()
   {
      apply = apply_;
      PStepCatcher* catcher = new PStepCatcher;
      catchSelect().append(catcher);
   }

   virtual ~PStepAfterNext() {}
   virtual void describeTo( String& target ) const { target = "Debugger::PStepAfterNext"; }

private:
   static void apply_( const PStep*, VMContext* ctx )
   {
      ctx->popCode();
      ctx->setBreakpointEvent();
   }
};

//========================================================================
// Command handler
//========================================================================

class CmdHandler
{
public:
   CmdHandler( Debugger* dbg, const String& name ):
      m_name(name),
      m_debugger(dbg)
   {}

   virtual ~CmdHandler(){}

   virtual void execute( VMContext* ctx, const String& ) = 0;
   const String& pdesc() const { return m_pdesc; }
   const String& desc() const { return m_desc; }
   const String& help() const { return m_help; }
   const String& name() const { return m_name; }

protected:
   String m_name;
   Debugger* m_debugger;

   String m_help;
   String m_pdesc;
   String m_desc;

   typedef std::vector<String> ParamList;

   // tokenize the parameters.
   void tokenize(const String& params, ParamList& list)
   {
      StringTokenizer tk(params,' ', true);
      String temp;
      while( tk.next(temp) )
      {
         list.push_back(temp);
      }
   }

private:
   // disable copy
   CmdHandler( const CmdHandler& ) {}
};


class CmdHelp: public CmdHandler
{
public:
   CmdHelp(Debugger* dbg):
      CmdHandler(dbg,"help")
   {
      m_pdesc = "[command]";
      m_desc = "Gives the list of commands or provide help on a command";
   }

   virtual ~CmdHelp(){}

   virtual void execute( VMContext*, const String& params )
   {
      if( params.empty() )
      {
         m_debugger->listCommands();
      }
      else
      {
         m_debugger->describe(params);
      }
   }
};


class CmdInsp: public CmdHandler
{
public:
   CmdInsp(Debugger* dbg):
      CmdHandler(dbg,"insp")
   {
      m_pdesc = "[var]";
      m_desc = "Inspects the given variable";
   }

   virtual ~CmdInsp(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      if( params.empty() )
      {
         m_debugger->writeLine("**: missing variable to be inspected");
      }
      else
      {
         try {
            Item* itm = ctx->resolveSymbol(params, false);
            Inspector insp(m_debugger->writer());
            insp.inspect(*itm);
         }
         catch( CodeError * e )
         {
            e->decref();
            m_debugger->writeLine("**: Symbol not found");
         }
      }
   }
};

class CmdQuit: public CmdHandler
{
public:
   CmdQuit(Debugger* dbg):
      CmdHandler(dbg,"quit")
   {
      m_desc = "Terminates the host program";
   }

   virtual ~CmdQuit(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->writeLine("**: terminating host process");
      ctx->process()->terminate();
      m_debugger->exitDebugger();
   }
};


class CmdCont: public CmdHandler
{
public:
   CmdCont(Debugger* dbg):
      CmdHandler(dbg,"cont")
   {
      m_desc = "Resumes the execution of the host program";
   }

   virtual ~CmdCont(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->writeLine("**: continuing");
      ctx->process()->setDebug(true);
      // force the processor to reload the debugged context
      ctx->setSwapEvent();
      m_debugger->exitDebugger();
   }
};


class CmdStep: public CmdHandler
{
public:
   CmdStep(Debugger* dbg):
      CmdHandler(dbg,"step")
   {
      m_desc = "Performs a single step in the code";
   }

   virtual ~CmdStep(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->writeLine( "**: single step" );
      ctx->setBreakpointEvent();
      m_debugger->exitDebugger();
   }
};


class CmdNext: public CmdHandler
{
public:
   CmdNext(Debugger* dbg):
      CmdHandler(dbg,"next")
   {
      m_desc = "Proceeds until a different source line is reached.";
   }

   virtual ~CmdNext(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->writeLine( "**: long step" );
      ctx->process()->addNegativeBreakpoint( ctx, ctx->currentCode().m_step );
      ctx->setSwapEvent();
      m_debugger->exitDebugger();
   }
};


class CmdEval: public CmdHandler
{
public:
   CmdEval(Debugger* dbg):
      CmdHandler(dbg,"eval")
   {
      m_desc = "Evaluates the rest of the line in in the current context";
      m_pdesc = "<cmd>";
   }

   virtual ~CmdEval(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      Stream* sinput = new StringStream( params );
      TextReader* reader = new TextReader(sinput);
      DynCompiler dynComp( ctx );

      try
      {
         SynTree* st = dynComp.compile( reader );
         ctx->pushData( FALCON_GC_HANDLE(st) );
         ctx->pushCodeWithUnrollPoint(m_debugger->m_stepPostEval);
         ctx->pushCode(st);
         // force to exit from debug evaluation
         ctx->clearEvents();
         ctx->setSwapEvent();
         ctx->process()->setDebug(false);
         m_debugger->exitDebugger();
      }
      catch( Error* err )
      {
         m_debugger->write("**!: ");
         m_debugger->write(err->describe(true));
         m_debugger->write("\n");
         err->decref();
      }
      reader->decref();
   }
};


class CmdSrc: public CmdHandler
{
public:
   CmdSrc(Debugger* dbg):
      CmdHandler(dbg,"src")
   {
      m_desc = "Locate current code in module and function";
   }

   virtual ~CmdSrc(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->printLoc(ctx);
      m_debugger->printCode(ctx);
   }
};


class CmdData: public CmdHandler
{
public:
   CmdData(Debugger* dbg):
      CmdHandler(dbg,"data")
   {
      m_desc = "Display the data in the stack (top N positions, 0 for all)";
      m_pdesc = "[N]";
   }

   virtual ~CmdData(){}

   virtual void execute( VMContext* ctx, const String& param )
   {
      if( param.empty() )
      {
         m_debugger->displayStack( ctx, 1 );
      }
      else
      {
         int64 depth;
         if( ! param.parseInt(depth) || depth < 0 )
         {
            m_debugger->writeLine( "**: invalid depth" );
         }
         else {
            m_debugger->displayStack( ctx, depth );
         }
      }
   }
};


class CmdDyns: public CmdHandler
{
public:
   CmdDyns(Debugger* dbg):
      CmdHandler(dbg,"dyns")
   {
      m_desc = "Display the data in the dynamic symbol stack (top N positions, 0 for all)";
      m_pdesc = "[N]";
   }

   virtual ~CmdDyns(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      if( params.empty() )
      {
         m_debugger->displayDyns( ctx, 1 );
      }
      else
      {
        int64 depth;
        if( ! params.parseInt(depth) || depth < 0 )
        {
           m_debugger->writeLine( "**: invalid depth" );
        }
        else {
           m_debugger->displayDyns( ctx, depth );
        }
      }
   }
};


class CmdCode: public CmdHandler
{
public:
   CmdCode(Debugger* dbg):
      CmdHandler(dbg,"code")
   {
      m_desc = "Display the data in the code stack (top N positions, 0 for all)";
      m_pdesc = "[N]";
   }

   virtual ~CmdCode(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      if( params.empty() )
      {
         m_debugger->displayCode( ctx, 1 );
      }
      else
      {
        int64 depth;
        if( ! params.parseInt(depth) || depth < 0 )
        {
           m_debugger->writeLine( "**: invalid depth" );
        }
        else {
           m_debugger->displayCode( ctx, depth );
        }
      }
   }
};


class CmdBack: public CmdHandler
{
public:
   CmdBack(Debugger* dbg):
      CmdHandler(dbg,"back")
   {
      m_desc = "Display the data in the call stack (top N positions, 0 for all)";
      m_pdesc = "[N]";
   }

   virtual ~CmdBack(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      if( params.empty() )
      {
         m_debugger->displayCall( ctx, 1 );
      }
      else
      {
        int64 depth;
        if( ! params.parseInt(depth) || depth < 0 )
        {
           m_debugger->writeLine( "**: invalid depth" );
        }
        else {
           m_debugger->displayCall( ctx, depth );
        }
      }
   }
};


class CmdBpl: public CmdHandler
{
public:
   CmdBpl(Debugger* dbg):
      CmdHandler(dbg,"bpl")
   {
      m_desc = "List the breakpoints.";
      m_pdesc = "";
   }

   virtual ~CmdBpl(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      Process* prc = ctx->process();

      class Rator: public Process::BreakpointEnumerator
      {
      public:
         Rator( Debugger* dbg ): m_dbg(dbg) {}
         virtual ~Rator() {}

         virtual void operator()(int id, bool bEnabled, const String& path, const String& name, int32 line, bool bTemp )
         {
            const char* mode = name == "" ? "P" : (bEnabled ? "E" : "D");
            m_dbg->write( String().N(id).A("[").A(mode).A("]:"));
            if( name != "" )
            {
               m_dbg->write(name);
            }
            else {
               m_dbg->write(path);
            }
            m_dbg->write(String(":").N(line));

            if( bTemp )
            {
               m_dbg->write("[T]");
            }
            m_dbg->writeLine("");
         }
      private:
         Debugger* m_dbg;
      }
      rator(m_debugger);

      prc->enumerateBreakpoints(rator);
   }
};

class CmdBpa: public CmdHandler
{
public:
   CmdBpa(Debugger* dbg):
      CmdHandler(dbg,"bpa")
   {
      m_desc = "Add a breakpoint";
      m_pdesc = "[module]:<line> [F]";
      m_help = "The the module is intended as a logical name of a module, unless the 'f' "
               "flag is added at the end of the command. In that case, the module name is intended "
               "as a URI, and the breakpoint is left pending until the given module is loaded."
               " If the module is not given, the current module is used.";
   }

   virtual ~CmdBpa(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      Process* prc = ctx->process();
      StringTokenizer pt(params, ' ');
      String bp, flag;
      pt.next(bp);
      pt.next(flag);

      // find the line.
      uint32 posComma = bp.rfind(':');
      int64 line = 0;
      if( posComma == String::npos || ! bp.subString(posComma+1).parseInt(line) || line <= 0 )
      {
         m_debugger->writeLine( "**: Invalid module:line format" );
         return;
      }

      int bpID = 0;
      if( posComma == 0 )
      {
         CallFrame& cf = ctx->currentFrame();
         if( cf.m_function != 0 && cf.m_function->module() != 0 )
         {
            bpID = prc->addBreakpoint( cf.m_function->module()->uri(), cf.m_function->module()->name(), line, false, true );
         }
         else {
            m_debugger->writeLine( "**: No current context" );
         }
      }
      else if( !flag.empty() || bp.find('/') )
      {
         bpID = prc->addBreakpoint( bp.subString(0,posComma), "", line, false, true );
      }

      m_debugger->writeLine( String("**: Added breakpoint ").N(bpID) );
   }

};


class CmdBpr: public CmdHandler
{
public:
   CmdBpr(Debugger* dbg):
      CmdHandler(dbg,"bpr")
   {
      m_desc = "Remove a breakpoint";
      m_pdesc = "[N]";
      m_help = "The parameter is the ID of the breakpoint, as reported by bpa and bpl commands";
   }

   virtual ~CmdBpr(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      Process* prc = ctx->process();
      int64 id = 0;
      if ( ! params.parseInt(id) || id <= 0 )
      {
         m_debugger->writeLine( "**: Invalid breakpoint ID" );
      }
      else if ( ! prc->removeBreakpoint((int) id) )
      {
         m_debugger->writeLine( "**: Breakpoint ID not found" );
      }
      else
      {
         m_debugger->writeLine( "**: Breakpoint removed" );
      }
   }
};


class CmdBpe: public CmdHandler
{
public:
   CmdBpe(Debugger* dbg):
      CmdHandler(dbg,"bpe")
   {
      m_desc = "Enables a breakpoint";
      m_pdesc = "[N]";
      m_help = "The parameter is the ID of the breakpoint, as reported by bpa and bpl commands";
   }

   virtual ~CmdBpe(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      Process* prc = ctx->process();
      int64 id = 0;
      if ( ! params.parseInt(id) || id <= 0 )
      {
         m_debugger->writeLine( "**: Invalid breakpoint ID" );
      }
      else if ( ! prc->enableBreakpoint((int) id, true) )
      {
         m_debugger->writeLine( "**: Breakpoint ID not found" );
      }
      else
      {
         m_debugger->writeLine( "**: Breakpoint enabled" );
      }
   }
};


class CmdBpd: public CmdHandler
{
public:
   CmdBpd(Debugger* dbg):
      CmdHandler(dbg,"bpd")
   {
      m_desc = "Disable a breakpoint";
      m_pdesc = "[N]";
      m_help = "The parameter is the ID of the breakpoint, as reported by bpa and bpl commands";
   }

   virtual ~CmdBpd(){}

   virtual void execute( VMContext* ctx, const String& params )
   {
      Process* prc = ctx->process();
      int64 id = 0;
      if ( ! params.parseInt(id) || id <= 0 )
      {
         m_debugger->writeLine( "**: Invalid breakpoint ID" );
      }
      else if ( ! prc->enableBreakpoint((int) id, false) )
      {
         m_debugger->writeLine( "**: Breakpoint ID not found" );
      }
      else
      {
         m_debugger->writeLine( "**: Breakpoint disabled" );
      }
   }
};



class CmdGlob: public CmdHandler
{
public:
   CmdGlob(Debugger* dbg):
      CmdHandler(dbg,"glob")
   {
      m_desc = "Display the globals table for the current module (if any)";
   }

   virtual ~CmdGlob(){}

   virtual void execute( VMContext* ctx, const String& )
   {
      m_debugger->displayGlobals( ctx );
   }
};


class Debugger::Private
{
public:
   typedef std::map<String,CmdHandler*> CmdMap;
   CmdMap m_commands;


   Private( Debugger* dbg )
   {
      addCommand(new CmdHelp(dbg));
      addCommand(new CmdQuit(dbg));

      addCommand(new CmdStep(dbg));
      addCommand(new CmdNext(dbg));
      addCommand(new CmdCont(dbg));
      addCommand(new CmdInsp(dbg));

      addCommand(new CmdEval(dbg));
      addCommand(new CmdSrc(dbg));

      addCommand(new CmdData(dbg));
      addCommand(new CmdDyns(dbg));
      addCommand(new CmdCode(dbg));
      addCommand(new CmdBack(dbg));
      addCommand(new CmdGlob(dbg));

      addCommand(new CmdBpl(dbg));
      addCommand(new CmdBpa(dbg));
      addCommand(new CmdBpr(dbg));
      addCommand(new CmdBpe(dbg));
      addCommand(new CmdBpd(dbg));
   }

   void addCommand( CmdHandler* cmd )
   {
      m_commands[cmd->name()] = cmd;
   }


   ~Private()
   {
      CmdMap::iterator iter = m_commands.begin();
      while( iter != m_commands.end() )
      {
         delete iter->second;
         ++iter;
      }
   }
};

//========================================================================
// Main Debugger
//========================================================================

Debugger::Debugger() :
    m_hello(true)
{
   StmtTry* t = new PStepPostEval;
   m_stepPostEval = t;
   m_stepAfterNext = new PStepAfterNext;
   _p = new Private(this);
}

Debugger::~Debugger()
{
   delete m_stepPostEval;
   delete m_stepAfterNext;
   delete _p;
}


void Debugger::listCommands() const
{
   m_tw->write( "*: Command list:\n" );
   Private::CmdMap::const_iterator iter = _p->m_commands.begin();
   while( iter != _p->m_commands.end() )
   {
      m_tw->write( "*: " );
      CmdHandler* h = iter->second;
      m_tw->write(h->name());
      m_tw->write(" ");
      m_tw->write(h->pdesc());
      m_tw->write(": ");
      m_tw->write(h->desc());
      m_tw->write("\n");
      ++iter;
   }
   m_tw->write("**: Entering an empty line repeats the previous command.\n");
}


void Debugger::describe(const String& cmd) const
{
   Private::CmdMap::const_iterator iter = _p->m_commands.find(cmd);
   if(iter == _p->m_commands.end())
   {
      m_tw->write("**: Command not found\n");
   }
   else {
      CmdHandler* h = iter->second;
      if( h->help().empty() )
      {
         m_tw->write("**:" + h->desc() +"\n");
      }
      else {
         m_tw->write("**:" + h->help() +"\n");
      }
   }
}


void Debugger::write(const String& str) const
{
   m_tw->write(str);
}


void Debugger::writeLine(const String& str) const
{
   m_tw->writeLine(str);
}



void Debugger::onBreak( Process* p, Processor*, VMContext* ctx )
{
   MESSAGE("Debugger::onBreak -- Debugger invoked.");

   bool addNil = false;
   // the result of the function Debugger.breakpoint() that called us
   if( ctx->topData().type() == 0xff )
   {
      addNil = true;
      ctx->popData();
   }

   // get the standard input stream.
   Stream* input = p->vm()->stdIn();
   Stream* output = p->vm()->stdErr();

   TextReader tr(input);
   TextWriter wr( output );
   m_tw = &wr;

   if( m_hello )
   {
      wr.writeLine( "*: Falcon debugger engaged, type \"help\" for help." );
      m_hello = false;
   }

   printCode( ctx );

   m_bActive = true;
   do
   {
      MESSAGE1("Debugger::onBreak -- Accepting command");
      wr.write( "*> " );
      wr.flush();

      String line;
      tr.readLine(line, 1024);

      parseCommand( line, ctx );
   }
   while( m_bActive );

   if( addNil )
   {
      // add a nil to simulate the exit of the funciton Debugger.breakpoint()
      ctx->pushData(Item());
   }
   // the result of the function.
   MESSAGE("Debugger::onBreak -- Exiting.");
}


void Debugger::parseCommand( const String& line1, VMContext* ctx )
{
   TRACE("Debugger::parseCommand -- parsing '%s'", line1.c_ize() );

   // whether to save this line or not
   bool save = true;

   String line = line1;
   line.trim();

   if( line == "" )
   {
      if (m_lastCommand != "")
      {
         Debugger::parseCommand( m_lastCommand, ctx);
      }
      // do nothing, will repeat prompt
      return;
   }

   String cmd, params;
   uint32 pos = line.find(' ');
   if( pos != String::npos )
   {
      cmd = line.subString(0,pos);
      params = line.subString(pos+1);
      params.trim();
   }
   else {
      cmd = line;
   }

   Private::CmdMap::iterator iter = _p->m_commands.find(cmd);
   if( iter != _p->m_commands.end() )
   {
      iter->second->execute(ctx, params);
   }
   else {
      m_tw->writeLine("*!: Unknown command");
      save = false;
   }

   if( save )
   {
      m_lastCommand = line;
   }

   MESSAGE1("Debugger::parseCommand -- return" );
}


void Debugger::printCode(VMContext* ctx)
{
   const PStep* step = ctx->nextStep();
   if( step->sr().line() == 0 )
   {
      m_tw->writeLine(step->describe());
   }
   else {
      const TreeStep* ts = static_cast<const TreeStep*>(step);
      m_tw->write(String("*: line ").N(step->sr().line()).A(": "));
      ts->render(m_tw,1);
   }
}


void Debugger::printLoc(VMContext* ctx)
{
   Function* func = ctx->currentFrame().m_function;
   if( func == 0 )
   {
      m_tw->writeLine("**: No current function context.");
   }
   else {
      m_tw->writeLine("**: in " + func->locate());
   }
}


void Debugger::displayStack( VMContext* ctx, int64 depth )
{
   m_tw->write( String("*: Data stack size ").N((int64)ctx->dataSize()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->dataSize()) || (depth > 0 && top < depth ) )
   {
      Item* item = &ctx->opcodeParam(top);
      Class* cls = 0;
      void* data = 0;
      item->forceClassInst(cls, data);

      String temp;
      cls->describe(data, temp, 3, 128);

      m_tw->write( String("*: ").H((uint64)item,true).A(" - ").N(top).A(": ").A(temp).A("\n") );
      ++top;
   }
}

void Debugger::displayDyns( VMContext* ctx, int64 depth )
{
   m_tw->write( String("*: Dyns stack size ").N((int64)ctx->dynsDepth()).A("\n") );
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

      m_tw->write( String("*: ").N(top).A(": ").A(dd->m_sym->name()).A("=(").H((uint64)item,true).A(") ").A(temp).A("\n") );
      ++top;
   }
}


void Debugger::displayCode( VMContext* ctx, int64 depth )
{
   m_tw->write( String("*: Code stack size ").N((int64)ctx->codeDepth()).A("\n") );
   int64 top = 0;
   while( (depth == 0 && top < ctx->codeDepth()) || (depth > 0 && top < depth ) )
   {
      CodeFrame* cf = ctx->codeAt(top);
      String temp;
      cf->m_step->describeTo(temp);
      m_tw->write( String("*: ").N(top).A(": ").A(temp).A("(").N(cf->m_seqId).A(")\n") );
      ++top;
   }
}

void Debugger::displayCall( VMContext* ctx, int64 depth )
{
   m_tw->write( String("*: Call stack size ").N((int64)ctx->callDepth()).A("\n") );
   TraceBack tb;
   ctx->fillTraceBack(&tb, true, depth);

   length_t size = tb.size();

   for ( length_t i = 0; i < size; ++i )
   {
      String str;
      TraceStep* ts = tb.at(i);
      ts->toString(str);
      m_tw->write( String("*: ").N(i).A(": ").A(str).A("\n") );
   }
}


void Debugger::displayGlobals( VMContext* ctx )
{
   Function* func = ctx->currentFrame().m_function;
   if( func->module() != 0 )
   {
      GlobalsMap& gmap = func->module()->globals();
      int64 size = gmap.size();
      m_tw->write( String("*: Global variables for module \"").A(func->module()->name()).A("\": ").N((int64)size).A("\n") );
      class Rator: public GlobalsMap::VariableEnumerator
      {
      public:
         Rator(TextWriter& tw) :m_tw(tw) {}
         virtual ~Rator() {}
         virtual void operator() ( const Symbol* sym, Item*& value )
         {
            m_tw.write( String("*: ").A(sym->name()).A(" (").H((uint64)value,true).A(")=").A(value->describe()).A("\n") );
         }

      private:
         TextWriter& m_tw;
      }
      rator(*m_tw);

      gmap.enumerate(rator);
   }
   else
   {
      m_tw->writeLine( "**: No current module\n" );
   }
}

}

/* end of debugger.cpp */

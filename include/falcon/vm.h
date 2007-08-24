/*
   FALCON - The Falcon Programming Language.
   FILE: flc_vm.h
   $Id: vm.h,v 1.56 2007/08/19 09:46:44 jonnymind Exp $

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 8 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef flc_VM_H
#define flc_VM_H

#include <falcon/symtab.h>
#include <falcon/symlist.h>
#include <falcon/item.h>
#include <falcon/stackframe.h>
#include <falcon/mempool.h>
#include <falcon/module.h>
#include <falcon/common.h>
#include <falcon/errhand.h>
#include <falcon/error.h>
#include <falcon/runtime.h>
#include <falcon/vmmaps.h>
#include <falcon/genericlist.h>
#include <falcon/carray.h>
#include <falcon/string.h>
#include <falcon/cobject.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/genericmap.h>
#include <falcon/genericlist.h>
#include <falcon/fassert.h>


#define FALCON_MAX_ATTRIBUTES 64
#define FALCON_VM_DFAULT_CHECK_LOOPS 1000

namespace Falcon {

class Runtime;
class VMachine;
class VMContext;
class PropertyTable;

typedef void (*tOpcodeHandler)( register VMachine *);

void ContextList_deletor( void * );

class FALCON_DYN_CLASS ContextList: public List
{
   friend void ContextList_deletor( void * );

public:
   ContextList()
   {}
};

class TryFrame
{
   uint32 m_pc;
   uint32 m_stackBase;
   uint32 m_frameBase;
   uint32 m_moduleId;

   friend class VMachine;
public:

   TryFrame( int32 pc, int32 sb, int32 fb, int32 modId ):
      m_pc( pc ),
      m_stackBase( sb ),
      m_frameBase( fb ),
      m_moduleId( modId )
      {}
};


void opcodeHandler_END ( register VMachine *vm );
void opcodeHandler_NOP ( register VMachine *vm );
void opcodeHandler_PSHN( register VMachine *vm );
void opcodeHandler_RET ( register VMachine *vm );
void opcodeHandler_RETA( register VMachine *vm );
void opcodeHandler_PTRY( register VMachine *vm );

   // Range 2: one parameter ops;
void opcodeHandler_LNIL( register VMachine *vm );
void opcodeHandler_RETV( register VMachine *vm );
void opcodeHandler_FORK( register VMachine *vm );
void opcodeHandler_BOOL( register VMachine *vm );
void opcodeHandler_GENA( register VMachine *vm );
void opcodeHandler_GEND( register VMachine *vm );
void opcodeHandler_PUSH( register VMachine *vm );
void opcodeHandler_PSHR( register VMachine *vm );
void opcodeHandler_POP ( register VMachine *vm );
void opcodeHandler_JMP ( register VMachine *vm );
void opcodeHandler_INC ( register VMachine *vm );
void opcodeHandler_DEC ( register VMachine *vm );
void opcodeHandler_NEG ( register VMachine *vm );
void opcodeHandler_NOT ( register VMachine *vm );
void opcodeHandler_TRAL( register VMachine *vm );
void opcodeHandler_IPOP( register VMachine *vm );
void opcodeHandler_XPOP( register VMachine *vm );
void opcodeHandler_GEOR( register VMachine *vm );
void opcodeHandler_TRY ( register VMachine *vm );
void opcodeHandler_JTRY( register VMachine *vm );
void opcodeHandler_RIS ( register VMachine *vm );
void opcodeHandler_BNOT( register VMachine *vm );
void opcodeHandler_NOTS( register VMachine *vm );
void opcodeHandler_PEEK( register VMachine *vm );

   // Range3: Double parameter ops;
void opcodeHandler_LD  ( register VMachine *vm );
void opcodeHandler_LDRF( register VMachine *vm );
void opcodeHandler_ADD ( register VMachine *vm );
void opcodeHandler_SUB ( register VMachine *vm );
void opcodeHandler_MUL ( register VMachine *vm );
void opcodeHandler_DIV ( register VMachine *vm );
void opcodeHandler_MOD ( register VMachine *vm );
void opcodeHandler_POW ( register VMachine *vm );
void opcodeHandler_ADDS( register VMachine *vm );
void opcodeHandler_SUBS( register VMachine *vm );
void opcodeHandler_MULS( register VMachine *vm );
void opcodeHandler_DIVS( register VMachine *vm );
void opcodeHandler_MODS( register VMachine *vm );
void opcodeHandler_BAND( register VMachine *vm );
void opcodeHandler_BOR ( register VMachine *vm );
void opcodeHandler_BXOR( register VMachine *vm );
void opcodeHandler_ANDS( register VMachine *vm );
void opcodeHandler_ORS ( register VMachine *vm );
void opcodeHandler_XORS( register VMachine *vm );
void opcodeHandler_GENR( register VMachine *vm );
void opcodeHandler_EQ  ( register VMachine *vm );
void opcodeHandler_NEQ ( register VMachine *vm );
void opcodeHandler_GT  ( register VMachine *vm );
void opcodeHandler_GE  ( register VMachine *vm );
void opcodeHandler_LT  ( register VMachine *vm );
void opcodeHandler_LE  ( register VMachine *vm );
void opcodeHandler_IFT ( register VMachine *vm );
void opcodeHandler_IFF ( register VMachine *vm );
void opcodeHandler_CALL( register VMachine *vm );
void opcodeHandler_INST( register VMachine *vm );
void opcodeHandler_ONCE( register VMachine *vm );
void opcodeHandler_LDV ( register VMachine *vm );
void opcodeHandler_LDP ( register VMachine *vm );
void opcodeHandler_TRAN( register VMachine *vm );
void opcodeHandler_UNPK( register VMachine *vm );
void opcodeHandler_SWCH( register VMachine *vm );
void opcodeHandler_HAS ( register VMachine *vm );
void opcodeHandler_HASN( register VMachine *vm );
void opcodeHandler_GIVE( register VMachine *vm );
void opcodeHandler_GIVN( register VMachine *vm );
void opcodeHandler_IN  ( register VMachine *vm );
void opcodeHandler_NOIN( register VMachine *vm );
void opcodeHandler_PROV( register VMachine *vm );
void opcodeHandler_SSTA( register VMachine *vm );
void opcodeHandler_SVAL( register VMachine *vm );
void opcodeHandler_STPS( register VMachine *vm );
void opcodeHandler_STVS( register VMachine *vm );
void opcodeHandler_LSTA( register VMachine *vm );
void opcodeHandler_LVAL( register VMachine *vm );
void opcodeHandler_AND ( register VMachine *vm );
void opcodeHandler_OR  ( register VMachine *vm );
void opcodeHandler_PASS( register VMachine *vm );
void opcodeHandler_PSIN( register VMachine *vm );

   // Range 4: ternary opcodes;
void opcodeHandler_STP ( register VMachine *vm );
void opcodeHandler_STV ( register VMachine *vm );
void opcodeHandler_LDVT( register VMachine *vm );
void opcodeHandler_LDPT( register VMachine *vm );
void opcodeHandler_STPR( register VMachine *vm );
void opcodeHandler_STVR( register VMachine *vm );
void opcodeHandler_TRAV( register VMachine *vm );

void opcodeHandler_SJMP( register VMachine *vm );
void opcodeHandler_FORI( register VMachine *vm );
void opcodeHandler_FORN( register VMachine *vm );

void opcodeHandler_SHL( register VMachine *vm );
void opcodeHandler_SHR( register VMachine *vm );
void opcodeHandler_SHLS( register VMachine *vm );
void opcodeHandler_SHRS( register VMachine *vm );
void opcodeHandler_LDVR( register VMachine *vm );
void opcodeHandler_LDPR( register VMachine *vm );
void opcodeHandler_POWS( register VMachine *vm );
void opcodeHandler_LSB( register VMachine *vm );
void opcodeHandler_UNPS( register VMachine *vm );
void opcodeHandler_SELE( register VMachine *vm );
void opcodeHandler_INDI( register VMachine *vm );
void opcodeHandler_STEX( register VMachine *vm );
void opcodeHandler_TRAC( register VMachine *vm );
void opcodeHandler_WRT( register VMachine *vm );


/** The Falcon virtual machine.

   Virtual machine is in charge to execute Falcon bytecode.

   The virtual machine execution model is quite flexible, and allow infinite execution,
   step-by-step execution, limited runs and interrupts (event rising and debug requests)
   from external code while running.

   To run, the Virtual Machine needs to be properly setup. However, this "setup steps" are
   thought as customization steps so that the VM user is able to configure it
   on its needs, and they are both simple and effective.

   Minimal configuration needed requires to add a Runtime object, which holds the lists of
   all non-main modules. It is then possible to execute a routine exported by any of the
   modules in the runtime, or to provide a "running module", or "main module" with the
   setup() method. Once the startup symbol, or running module (or both) are set, the
   it is possible to call run().

   \note It is possible that this interface will be simplified in the future so that only one
      method is used to set up the runtime and the start symbol or main module.


   \note This class implements the error handler interface because it may act as an
      error handler for the embedding components: compiler, assembler, module loader
      and runtime may all be embedded in a VM that may act as a referee for the
      script to be i.e. evaluating correctly a piece of falcon source code.
*/
class FALCON_DYN_CLASS VMachine: public ErrorHandler
{

public:
   typedef enum {
      eventNone,
      eventQuit,
      eventRisen,
      eventReturn,
      eventYield,
      eventWait,
      eventSuspend,
      eventSingleStep,
      eventOpLimit,
      eventSleep,
      eventInterrupt
   } tEvent;

protected:
   /** Structure used to save the vardefs during class link */
   class VarDefMod: public BaseAlloc
   {
   public:
      VarDef *vd;
      int16 modId;
   };

   /** Currently executed symbol.
      May be 0 if the startmodule has not a "__main__" symbol;
      this should be impossible when things are set up properly
      \todo, make always nonzeor
   */
   Symbol* m_symbol;
   /** Position of the module that contains the current symbol in the globals vector. */
   uint16 m_moduleId;

   Item m_regA;
   Item m_regB;
   Item m_regS1;
   Item m_regS2;

   /** Space for immediate operands. */
   Item m_imm[4];

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;
   bool m_bhasStandardStreams;

   /** Current stack.
      Selected from one of the contexts (coroutines).
   */
   ItemVector *m_stack;

   /** Number of opcodes that the current coroutine has performed. */
   uint32 m_opCount;

   /** Maximum operations that can be performed */
   uint32 m_opLimit;

   /** Timeout for GC collection */
   uint32 m_loopsGC;
   /** Timeout for Context checks */
   uint32 m_loopsContext;
   /** Timeout for Callbacks */
   uint32 m_loopsCallback;

   /** True for single stepping */
   bool m_bSingleStep;

   uint32 m_opNextGC;
   uint32 m_opNextContext;
   uint32 m_opNextCallback;

   /** Smaller check in op loops. */
   uint32 m_opNextCheck;

   /** Variables are organized in this way
      Each modules has a vector of varibles in which each position is occupied by
      the item that represents the current value of a given symbol. This vector is
      the "module globals vector". Each module is given an ID from 1 to N that
      represents the position of the module in the module list.

      m_globals is filled during the link phase.
   */
   GlobalsVector m_globals;

   uint32 m_stackBase;

   /** Currently executed code.
      It's the code from m_symbol->module()->code(),
      fetched here for brevity. Possibly removed.
   */
   byte *m_code;

   /** Program counter register.
      Current execution point in current code.
   */
   uint32 m_pc;

   /** Parameter count in BOM operations.
      Normally, parameter count is set in stack frames, but BOM calls are faster and
      don't build a frame. This variable is used to record the parameter count for
      BOM calls. Don't use it elsewhere.
   */
   int32 m_bomParams;

   /** Next program counter register.
      This is the next instruction the VM has to execute.
      Call returns and jumps can easily modify the VM execution flow by
      changing this value, that is normally set just to m_pc + the lenght
      of the current instruction.
   */
   uint32 m_pc_next;

   /** Context currently being executed. */
   VMContext *m_currentContext;

   /** Modules held in this VM */
   ModuleVector m_modules;

   /** Ready to run contexts. */
   ContextList m_contexts;
   /** Contexts willing to sleep for a while */
   ContextList m_sleepingContexts;
   /** Wether or not to allow a VM hostile takeover of the current context. */
   bool m_allowYield;



   /** Raised error.
      If the VM receives an error (i.e. via raiseError() ), this variable gets the
      pointer to the error instance. The error is passed to the handler, and then the
      VM terminates.
      The VM can either terminate execution cleanly (on empty stack return or via END opcode)
      or because of an error being placed in this variable. So, if the VM terminates with
      error this variable will be filled with a proper value.
   */

   Error *m_error;

   /** Error handler ownership.
      Signals if this VM owns its error handler (m_errhand). Some errorhandler may be
      provided by the application and shared among many objects, including the application
      itself. Usually, the VM will create its own error handler and dispose of it, but
      if provided externally, the error handler may be left undeleted at VM destruction.
   */
   bool m_bOwnErrorHandler;

   /** Error handler.
      By default set to a DefaultErrorHandler with an unowned copy of m_stdErr Falcon Stream.
      The error handler may be owned or not (destroyed at VM destruction or not) depending on
      the m_bOwnErrorHandler setting.
   */
   ErrorHandler *m_errhand;

   MemPool *m_memPool;

   /** Stack of global variables for the current module.
      This is a vector (or stack) of variables that represent the globals for the active module.

      This pointer points to one of the m_globals vector; it is also stored here to spare
      a possibly frequent access to the nth member of m_globals.

      As they are allocate either in the original symbol or by the memory pool manager,
      the void * in the stack are \b not owned by this vector, which just refers them and
      puts them in relation with the global symbol ID stored in the module opcodes.

      As the module list cannot be changed after the link phase, this pointer is granted to
      stay valid for the whole VM run activity; if this ever changes (i.e. if the some ability
      to load the modules at runtime is provided) this pointer must be removed.
   */
   ItemVector *m_currentGlobals;
   ItemVector m_superGlobals;

   /** Opcode hanlder function calls. */
   tOpcodeHandler *m_opHandlers;

   /** Events that are required */
   tEvent m_event;

   /** List of cached TRY positions */
   List *m_trypos;

   /** Switch to suspend terminated modules.
      With this switch on, it is possible to not discard global variables when the execution of
      a module terminates. In this case, the global variables are stored in the globals ID vector,
      and can be reused later on without the need to be re-created.
   */
   bool m_onDoneSuspend;

   /** Switch to suspend return instead waiting.
      \see returnOnSleep()
   */
   bool m_sleepAsRequests;

   /** Map of global symbols (and the item they are connected to).
      Each item of the map contains a Symbol * and an ID that allows to
   */
   SymModuleMap m_globalSyms;

   void yield( numeric seconds );
   void putAtSleep( VMContext *ctx, numeric secs );
   void rotateContext();

   numeric m_yieldTime;

   bool linkSubClass( uint32 modId, const Symbol *clssym, Map &props, uint64 &attribs );
   /** Count of the attributes during the link phase.*/
   int32 m_attributeCount;

   /** Passes current frame to the called item.
         Used by the PASS opcode, this element removes the local variables and
         prepares other local variables for the called item, but does not alter
         the stack frame. As the first ret opcode is hit, control returns to the
         caller of the item that generated this kind of pass; in other words,
         the Falcon function using callItem this way is effectively canceled as if
         never called (you won't see it even in a debug stack frame).
   */
   bool callItemPass( const Item &callable );

   /** Passes parameters of this function to the called item.
      When callItem is called in this way, a normal stack frame is generated,
      the function parameters just being copied into a new stack frame. This is
      a shortcut to a normal call, preceded by pushing of all the original parameters.
   */
   bool callItemPassIn( const Item &callable );

   friend class VMContext;
   friend class VMSemaphore;

   /** Subscribed services map.
      Services are allocated in the respective module.
      Currently, there's no way to remove a module from a VM once it's linked,
      so there's no need for a special de-subscription mechanism.

      Of course, if/when an unlink method is added, the module should also de-subscribe
      its services.
   */
   Map m_services;

   /** Utility for switch opcode.
      Just pretend it's not here.
   */
   bool seekInteger( int64 num, byte *base, uint16 size, uint32 &landing ) const;

   /** Utility for switch opcode.
      Just pretend it's not here.
   */
   bool seekInRange( int64 num, byte *base, uint16 size, uint32 &landing ) const;

   /** Utility for switch opcode.
      Just pretend it's not here.
   */
   bool seekString( const String *str, byte *base, uint16 size, uint32 &landing ) const;

   /** Utility for switch opcode.
      Just pretend it's not here.
   */
   bool seekItem( const Item *item, byte *base, uint16 size, uint32 &landing );

    /** Utility for select opcode.
      Just pretend it's not here.
   */
   bool seekItemClass( const Item *obj, byte *base, uint16 size, uint32 &landing ) const;


   void internal_construct();

   Item *parseSquareAccessor( const Item &accessed, String &accessor ) const;

   /** Returns the next NTD32 parameter, advancing the pointer to the next instruction */
   int32 getNextNTD32()
   {
      register int32 ret = endianInt32(*reinterpret_cast<int32 *>( m_code + m_pc_next ) );
      m_pc_next += sizeof( int32 );
      return ret;
   }

   /** Returns the next NTD64 parameter, advancing the pointer to the next instruction */
   int64 getNextNTD64()
   {
      register int64 ret = endianInt64(*reinterpret_cast<int64 *>( m_code + m_pc_next ) );
      m_pc_next += sizeof( int64 );
      return ret;
   }

   /** Gets the nth parameter. */
   Item *getOpcodeParam( register uint32 bc_pos );

public:

   /** Types of item calls.
      This type must be passed to callItem function to determine the mode by which an object
      in the VM is called.
      \see callItem()
   */
   typedef enum {
      /** Normal call mode.
         When an item is called in this way, as a return PCODE resets the stack at the same
         level as it was before the call, the VM returns. In other words, when the called
         function returns or otherwise terminates, the control returns to the caller.
      */
      e_callNormal,

      /** Builds a VM compliant return frame.
         As a return instruction is hit in the called item, the control is passed to the
         VM item that was active before callItem(). It's the standard way falcon functions
         are called.
      */
      e_callFrame,

      /** Calls an object and return control but respect self object.
          This is the same as callNormal mode, except for the fact that self and sender
          are not updated. This is used eventually by user
          functions that wish to be seen just as extensions of methods they are called
          from.
      */
      e_callInst,

      /** Calls an object and continue, but respect self object.
          This is the same as callFrame mode, except for the fact that self and sender
          are not updated. This is used by the INST opcode, that needs to emulate a
          VM internal call, but respecting the SELF object that may have been created
          in the meanwhile.
      */
      e_callInstFrame

   } e_callMode;

   /** Initialize VM from subclasses.
      Subclasses willing to provide their own initialization routine,
      or code wishing to configure the machine, may use this constructor
      to prevent the call of the init() routine of this class, which fills
      some configurable component of the machine.

      \param initItems false to prevent creation of base items
      \see init()
   */
   VMachine( bool initItems );


   /** Creates the virtual machine.
      The default constructor calss the virtual method init().
      Subclasses may overload it to setup personalized VMs. \see init().
   */
   VMachine();

   /** Destroys the virtual machine.
      It destroys the mempool manager, that causes also memory cleaning (complete fast garbage collecting)
      and memory pool destruction.
      It does not destroys the runtime, that must be destroyed by the caller. In the future, a reference
      count may be added both to the mempool and the runtime.
   */
   virtual ~VMachine();

   /** Initialize the virutal machine.
      Setups some configurable items that are used by the machine.

      They are namely:

      - m_memPool: the memory manager manager for this VM. The default
        is to use the Falcon::MemPool() manager.
      - m_stdIn: standard input. The default is to use an instance returned by
        stdInputStream(), which interfaces a localized text version of
        process stdin.
      - m_stdOut: standard output. The default is to use an instance returned by
        stdOutputStream(), which interfaces a localized text version of
        process stdout.
      - m_stdErr: standard error. The default is to use an instance returned by
        stdErrorStream(), which interfaces a localized text version of
        process stderr.
      - m_errhand: error handler. The default is to create an instance of
        DefaultErrorHandler built on the given m_stdErr (without ownership, as
        the VM destroys m_stdErr on exit).

      The subclass should set up its own items, if the default ones are not
      suitable, and then call the base class init(). The base init will
      not overwrite those items if they are valorized, but it may configure
      them to set up them properly, so it should always be called even if
      all the items are set by the subclass.

      \see VMachine( bool )
   */
   void init();

   bool link( Runtime *rt );
   bool link( Module *module );

   CoreClass *linkClass( uint32 modId, Symbol *clsym );

   /** Prepares a routine.
      The launch() method calls prepare() and run() in succession.
      A debugging environment should call prepare() and then singleStep()
      iteratively.
   */
   bool prepare( const String &startSym, uint32 paramCount = 0 );


   /** Launches the "__main__" symbol.
      This is a proxy call to launch( const String &);
      \return true if execution is successful, false otherwise.
   */
   bool launch() { return launch( "__main__" ); }

   /** Launches a routine.

      This methods prepares the execution environment of the script by allocating the items that
      represents the global variables in the various modules.
      This step is called "realization" of the Runtime, as it provides a set of "real"
      (in the sense of effective) values for the items that are drawn in the library. This may be a
      relatively long step.

      If the VM had been already used to launch a routine, the status and memory are cleaned and
      everything is returned to the startup state.

      Then the routine whose name is given is searched in the top module (the one that has been added last
      to the runtime). If not found there, it is searched in the exported symbols of the whole runtime.
      If not given, the __main__ symbol of the top module is executed. If the required symbol cannot be
      found an error is risen and false is returned.

      The routine returns true if execution is successful; the VM may return because the routine terminates,
      because of an explicit END opcode and because execution limit (single step or limited step count) is
      matched. Also, uncaught exceptions or explicit requests for termination may cause the VM to return.

      It is possible to know if the execution is complete by looking at the last VM event with the
      lastEvent() method: if none or quit, then the execution is just terminated, else an event is set
      (i.e. tSuspend).

      \param startSym the name of a routine to be executed.
      \param paramCount Number of parameters that have been pushed in the stack as parameters.
      \return true if execution is successful, false otherwise.
   */
   bool launch( const String &startSym, uint32 paramCount = 0 )
   {
      if ( prepare( startSym, paramCount ) ) {
         run();
         return true;
      }
      return false;
   }


   /** Virtual machine main loop.
      This is the method that is responsible for the main virtual
      machine loop. The VM can be configured to run a limited number
      of steps (or even one); there are also several other settings
      that may affect the run behavior.

      Usually, the user program won't call this; the launch() method sets up the
      execution environment and then calls run(). Calling this method is useful
      to continue a pending execution (including single step execution).

      If a symbol name is provided, then the symbol is executed retaining all the current
      status. The symbol is searched in the locals of the top module, and then in the
      global of the runtime eaxtly as for launch(). This is useful, in example, to provide
      callback entry points in scripts that must maintain their execution status between
      calls. In this way, each run maintain the previous status but grants execution of
      different entry points; also this is quite faster than having to enter the realization
      phase.

      An error will be risen if launch has not been previously called and the routine name
      is not provided.
   */
   void run();


   /** Raises a prebuilt error.
      This method raises an error as-is, without creating a traceback nor setting a
      context for it.
      \param err the pre-built error to be raised.
   */
   void raiseError( Error *err );

   /** Raises an error generated at runtime.
      This method raises an error and fill its running context: execution line, PC, symbol
      and module, other than the traceback.

      \param err the pre-built error to be raised.
   */
   void raiseRTError( Error *err );

   /** Raises an error coming from a module.

      This version of the method raises an error that has been created with its own
      execution context (module, symbol, line and PC). However, the traceback gets
      filled by the VM.

      Usually, extension modules will want to use this method as they may be interested
      to place their own module ID and C source line in the error context.

      \param err the pre-built error to be raised.
   */
   void raiseModError( Error *err );

   /** Raises a VM specific error.

      If a context is actually active, (i.e. if we are running a symbol),
      the context and the traceback are set accordingly; The class of the
      error is set to CodeError, and the origin is set to VM.

      This function is used by opcodes and other VM related stuff, and should
      not be used otherwise.

      \note As this function is meant to raise VM errors, it raises HARD errors, that is
      errors that can't be stopped by scripts.

      \param code the error code to be risen.
      \param line the script line where the error happened. If zero, it will be
         determined by the context, if we're running one.

   */
   void raiseError( int code, int32 line = 0 ) { raiseError( code, "", line ); }

   /** Raises a VM specific error.
      If a context is actually active, (i.e. if we are running a symbol),
      the context and the traceback are set accordingly; The class of the
      error is set to CodeError, and the origin is set to VM.

      This function is used by opcodes and other VM related stuff, and should
      not be used otherwise.

      \note As this function is meant to raise VM errors, it raises HARD errors, that is
      errors that can't be stopped by scripts.

      \param code the error code to be risen.
      \param line the script line where the error happened. If zero, it will be
         determined by the context, if we're running one.
      \param expl an eventual explanation of the error conditions.
      \see errorHandler()
   */
   void raiseError( int code, const String &expl, int32 line = 0 );


   /** Fills an error traceback with the current VM traceback. */
   void fillErrorTraceback( Error &error );

   /** Fills an error with current VM execution context and traceback.
   */
   void fillErrorContext( Error *err, bool filltb = true );


   ItemVector &superGlobals() { return m_superGlobals; }
   const ItemVector &superGlobals() const { return m_superGlobals; }


   /** Returns the current stack as a reference. */
   ItemVector &currentStack() { return *m_stack; }

   /** Returns the current stack as a reference (const version). */
   const ItemVector &currentStack() const { return *m_stack; }

   /** Returns a reference to the nth item in the current stack. */
   Item &stackItem( uint32 pos ) { return *(Item *) currentStack().at( pos ); }

   /** Returns a reference to the nth item in the current stack (const version). */
   const Item &stackItem( uint32 pos ) const { return *(Item *)currentStack().at(pos); }

   /** Returns the current module global variables vector. */
   ItemVector &currentGlobals() { return *m_currentGlobals; }

   /** Returns the current module global variables vector (const version). */
   const ItemVector &currentGlobals() const { return *m_currentGlobals; }

   /** Returns a reference to the nth item in the current module global variables vector. */
   Item &moduleItem( uint32 pos ) { return currentGlobals().itemAt( pos ); }
   /** Returns a reference to the nth item in the current module global variables vector. */
   Item &moduleItem( int16 modId, uint32 pos ) { return m_globals.vat(modId).itemAt( pos ); }

   /** Returns a reference to the nth item in the current module global variables vector (const version). */
   const Item &moduleItem( uint32 pos ) const { return currentGlobals().itemAt( pos ); }

   /** Returns the module in which the execution is currently taking place. */
   Module *currentModule() const { return m_modules.moduleAt( m_moduleId ); }

   /** Returns the globals vector of nth module. */
   ItemVector &globalsOfModule( uint32 id ) { return m_globals.vat( id ); }

   /** Returns the globals vector of nth module (const version). */
   const ItemVector &globalsOfModule( uint32 id ) const { return m_globals.vat( id ); }

   /** Returns a reference to the nth item in the superglobal vector. */
   Item &globalItem( uint32 pos ) { return superGlobals().itemAt( pos ); }

   /** Returns a reference to the nth item in the superglobal vector (const version). */
   const Item &globalItem( uint32 pos ) const { return superGlobals().itemAt( pos ); }

   /** Returns the count of linked modules */
   uint32 moduleListSize() const { return m_globals.size(); }

   /** Unlink a certain count of modules, so that the count of linked modules is reduced.
      This function unloads from the virtual machine all the modules that have
      been loaded after the nth one (given as parameter). The modules are decreffed,
      and may eventually be destroyed.

	  To restore the VM to a previous link status, record the return value of moduleListSize()
	  and apply it to the unlinkUpTo() method. In example:

	  \code
	  vm->link( rtl_module );
	  uint32 modCount = vm->moduleListSize();  // suppose 3, modules 0, 1 and 2

     my_script_runtime->...  // load the script
	  vm->link( my_script_runtime );
     vm->launch();

	  //reset the VM so that it keeps rtl
	  vm->unlinkUpTo( modCount );  // removes everything above module 2 (from 3 to size).
	  \endcode

      \param count number of modules to be left in the VM.
   */
   bool unlinkUpTo( uint32 count );

   const Module *getModule( int16 modId ) const {
      return m_modules.moduleAt( modId );
   }

   /** Sets the error handler.
      If there is a previously owned error handler in this VM, it is destroyed now.
      \param em the error handler to be set
      \param own true to allow VM to destroy this handler when disposing of it, false if the error handler is shared
                 with the application.
   */
   void errorHandler( ErrorHandler *em, bool own = false );
   ErrorHandler *errorHandler() const { return m_errhand; }

   /** Returns the parameter count for the current function.
      \note Calling this when the VM is not engaged in executing a function will crash.
      \return parameter count for the current function.
   */
   int32 paramCount() const {
      return ((StackFrame *)m_stack->at( m_stackBase - VM_FRAME_SPACE ) )->m_param_count;
   }

   /** Returns the parameter count in the current BOM handler.
      Falcon Basic Object Model handler are those methods that handle
      non-object items in "self" register (s1).

      As their call is handled specially, they have a special mean to
      access their stack elements (read: their parameters).

      \note Calling this from a non-bom handler will crash.

      \return number of parameters passed to the current BOM handler
   */
   int32 bomParamCount() const {
      return m_bomParams;
   }

   /** Returns the nth BOM handler parameter.
      Falcon Basic Object Model handler are those methods that handle
      non-object items in "self" register (s1).

      As their call is handled specially, they have a special mean to
      access their stack elements (read: their parameters).

      \param itemId the number of the parameter accessed, 0 based.
      \return a valid pointer to the (dereferenced) parameter or 0 if itemId is invalid.
      \note Calling this from a non-bom handler will crash.
   */
   Item *bomParam( uint32 itemId )
   {
      register uint32 params = bomParamCount();
      if ( itemId >= params ) return 0;
      return stackItem( m_stack->size() - params + itemId ).dereference();
   }

   /** Returns the nth paramter passed to the VM.
      The count is 0 based (0 is the first parameter).
      If the parameter exists, a pointer to the Item holding the
      parameter will be returned. If the item is a reference,
      the referenced item is returned instead (i.e. the parameter
      is dereferenced before the return).

      The pointer may be modified by the caller, but this will usually
      have no effect in the calling program unless the parameter has been
      passed by reference.

      \param itemId the number of the parameter accessed, 0 based.
      \return a valid pointer to the (dereferenced) parameter or 0 if itemId is invalid.
      \see isParamByRef
   */
   const Item *param( uint32 itemId ) const
   {
      register uint32 params = paramCount();
      if ( itemId >= params ) return 0;
      return stackItem( m_stackBase - params - VM_FRAME_SPACE + itemId ).dereference();
   }

   /** Returns the nth paramter passed to the VM.

      This is just the noncost version.

      The count is 0 based (0 is the first parameter).
      If the parameter exists, a pointer to the Item holding the
      parameter will be returned. If the item is a reference,
      the referenced item is returned instead (i.e. the parameter
      is dereferenced before the return).

      The pointer may be modified by the caller, but this will usually
      have no effect in the calling program unless the parameter has been
      passed by reference.

      \param itemId the number of the parameter accessed, 0 based.
      \return a valid pointer to the (dereferenced) parameter or 0 if itemId is invalid.
      \see isParamByRef
   */
   Item *param( uint32 itemId )
   {
      register uint32 params = paramCount();
      if ( itemId >= params ) return 0;
      return stackItem( m_stackBase - params - VM_FRAME_SPACE + itemId ).dereference();
   }


   /** Returns the nth local item.
      The first variable in the local context is numbered 0.
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   const Item *local( uint32 itemId ) const
   {
      return stackItem( m_stackBase + itemId ).dereference();
   }

   /** Returns the nth local item.
      This is just the non-const version.
      The first variable in the local context is numbered 0.
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   Item *local( uint32 itemId )
   {
      return stackItem( m_stackBase + itemId ).dereference();
   }

   /** Returns true if the nth element of the current function has been passed by reference.
      \param itemId the number of the parameter accessed, 0 based.
      \return true if the parameter exists and has been passed by reference, false otherwise
   */
   bool isParamByRef( uint32 itemId ) const
   {
      register uint32 params = paramCount();
      if ( itemId >= params ) return false;
      return stackItem( m_stackBase - params - VM_FRAME_SPACE + itemId ).type() == FLC_ITEM_REFERENCE;
   }

   const Item &regA() const { return m_regA; }
   Item &regA() { return m_regA; }
   const Item &regB() const { return m_regB; }
   Item &regB() { return m_regB; }

   const Item &self() const { return m_regS1; }
   Item &self() { return m_regS1; }
   const Item &sender() const { return m_regS2; }
   Item &sender() { return m_regS2; }

   void requestQuit() { m_event = eventQuit; }
   void requestSuspend() { m_event = eventSuspend; }
   tEvent lastEvent() const { return m_event; }

   void resume()
   {
      m_event = eventNone;
      retnil();
      run();
   }

   void resume( const Item &returned )
   {
      m_event = eventNone;
      retval( returned );
      run();
   }

   MemPool *memPool() const  { return m_memPool; }

   void retval( int32 val ) {
       m_regA.setInteger( (int64) val );
   }

   void retval( int64 val ) {
       m_regA.setInteger( val );
   }

   void retval( numeric val ) {
       m_regA.setNumeric( val );
   }

   void retval( const Item &val ) {
       m_regA = val;
   }

   /*void retval( String *cs ) {
      m_regA.setString(cs);
   }*/

   /** Returns a garbageable string.
      The string must have been allocated with MemPool::allocString() or there will be a leack
      (and an assert in debug).

      \note to put a nongarbage string in the VM use regA() accessor, but you must know what you're doing.
   */
   void retval( GarbageString *cs )
   {
      m_regA.setString(cs);
   }

   void retval( CoreArray *ca ) {
      m_regA.setArray(ca);
   }

   void retval( CoreDict *cd ) {
      m_regA.setDict( cd );
   }

   void retval( CoreObject *co ) {
      m_regA.setObject(co);
   }

   void retval( CoreClass *cc ) {
      m_regA.setClass(cc);
   }

   void retval( const String &cs ) {
      GarbageString *cs1 = new GarbageString( this, cs );
      cs1->bufferize();
      m_regA.setString( cs1 );
   }

   void retnil() { m_regA.setNil();}

   Symbol *currentSymbol() const { return m_symbol; }
   uint32 programCounter() const { return m_pc; }

   const SymModule *findGlobalSymbol( const String &str ) const;

   Item *findGlobalItem( const String &name ) const
   {
      const SymModule *sm = findGlobalSymbol( name );
      if ( sm == 0 ) return 0;
      return m_globals.vat( sm->moduleId() ).itemAt( sm->symbolId() ).dereference();
   }


   /** Returns the value of a local variable.
      This function searces for a variable in the local context, and eventually in the
      global context if not locally found.
      The function also parses accessors as the "." dot operator or the [] array access
      operator, but it doesn't support range accessors or function calls.

      The returned pointer is the place where the variable is physically stored.

      If the variable cannot be found, 0 is returned.
      \param name the variable to be found
      \return the physical pointer to the variable storage, or 0 on failure.
   */
   Item *findLocalVariable( const String &name ) const;

   /** Returns the value of a local variable.
      Similar to findLocalVariable(), but will return only the item coresponding to the
      named symbol. The symName parameter must be already trimmed and corespond exactly
      to the name of a variable in the local context.

      If the symbol is not found in the local symbol table, accessible global tables
      are searched in order of visibility.

      If the variable cannot be found, 0 is returned.
      \param symName the symbol name to be found
      \return the physical pointer to the variable storage, or 0 on failure.
   */
   Item *findLocalSymbolItem( const String &symName ) const;

   /** Calls an item.
      The item may contain any valid callable object:
      - An external (C/C++) function.
      - A falcon function.
      - A method
      - A class

      External functions are immediately called, the flow of the VM being
      interrupted.

      Falcon function and method calls prepare this vm to execute the first
      instruction of the item, pushing a stack frame that allows RET and similar
      pcodes to return to the VM instruction after the one that was currently
      being executed. Control may or may not return immediately to the caller;
      if e_callNormal, e_callInst or e_callPassIn modes are used, the function
      returns only after the called item returns, else the function returns
      immediately and the vm is readied to continue execution from the new
      position.

      Class calls actually searches for the class constructor, and call that one.
      If the called class has no constructor, the function returns true but actually
      does nothing.

      Before calling this function, enough parameters must be pushed in the stack.
      The paramCount parameter must be smaller or equal to the size of the stack,
      or an unblockable error will be raised.

      \param callable the item to be called.
      \param paramCount the number of elements in the stack to be considered parameters.
      \param mode the item call mode.

      \return false if the item is not callable, true if the item is called.
   */
   bool callItem( const Item &callable, int32 paramCount, e_callMode mode=e_callNormal );
   void pushParameter( const Item &item ) { m_stack->push( const_cast< Item *>(&item) ); }

   byte operandType( byte opNum ) const {
      return m_code[m_pc + 1 + opNum];
   }

   /** True if the VM is allowed to execute a context switch. */
   bool allowYield() { return m_allowYield; }
   /** Change turnover mode. */
   void allowYield( bool mode ) { m_allowYield = mode; }
   void yieldRequest( numeric time ) { m_event = eventYield; m_yieldTime = time; }


   const ContextList *getCtxList() const { return &m_contexts; }
   const ContextList *getSleepingList() const { return &m_sleepingContexts; }

   void suspendOnDone( bool sus=true ) { m_onDoneSuspend = sus; }
   bool suspendOnDone() const { return m_onDoneSuspend; }

   /** Return from the last called subroutine.
      Usually used internally by the opcodes of the VM.
   */
   void callReturn();

   /** Converts an item into a string.
      The string is NOT added to the garbage collecting system,
      so it may be disposed freely by the caller.
      This translates into a direct call to Item::toString(), unless the
      item is an object.
      In that case, if the item provides a toString() method, then that method is called
      passing the "format" parameter (only if it's not empty).

      \param itm a pointer to the item that must be represented as a string,
      \param format a format that will be eventually sent to the "toString" method of the object.
      \param target the output representation of the string
   */
   void itemToString( String &target, const Item *itm, const String &format );

   void itemToString( String &target, const Item *itm )
   {
      itemToString( target, itm, "" );
   }


   /** Creates the template for a core class.
      \param modId the module id for which this class is being created.
      \param pt a ClassDef wise property table containing properties definition.
      \return a newly created property table ready to be added to a core class.
   */
   PropertyTable *createClassTemplate( int modId, const Map &pt );

   /** Publish a service.
      Will raise an error and return false if the service is already published.
      \param svr the service to be published on this VM.
      \return true if the service can be registered, false (with error risal) if another service had that name.
   */
   bool publishService( Service *svr );


   /** Queries the VM for a published service.
      If a service with the given name has been published, the VM will return it;
      otherwise returns 0.
      \param name the service to be published on this VM.
      \return the required service or 0.
   */
   Service *getService( const String &name );

   /** Returns the module ID of a given module.
      The function is quite inefficient now, so use sparcely.
      Currently, is used to deserialize callable items.
      \param modName the module name as it has been inserted in the VM
      \return The module id if the name can be found, or -1 if not found.
   */
   int16 getModuleId( const String modName );

   /** Gets the standard input stream associated with this VM.
      VM may offer this stream to RTL and module functions wishing to
      get some input (as, i.e., input() ).

      By default, it is pointing
      to a correctly internationalized verisons of text-oriented Stream
      Transcoders. The transcoder is selected depending on the system
      environmental conditions, but there is no particular reason
      to use a transcoder instead of a direct stream, if this vm and
      functions using the standard streams abstractions held in the VM
      are willingt to operate on binary streams or on streams with
      different encodings.

      \return current incarnation of standard input stream.
   */

   Stream *stdIn() const { return m_stdIn; }

   /** Gets the standard input stream associated with this VM.
      VM may offer this stream to RTL and module functions wishing to
      write generic output, as print, printl and inspect.

      By default, it is pointing
      to a correctly internationalized verisons of text-oriented Stream
      Transcoders. The transcoder is selected depending on the system
      environmental conditions, but there is no particular reason
      to use a transcoder instead of a direct stream, if this vm and
      functions using the standard streams abstractions held in the VM
      are willingt to operate on binary streams or on streams with
      different encodings.

      \return current incarnation of standard output stream.
   */
   Stream *stdOut() const { return m_stdOut; }

   /** Gets the standard input stream associated with this VM.
      VM may offer this stream to RTL and module functions wishing to print
      error messages.

      By default, it is pointing
      to a correctly internationalized verisons of text-oriented Stream
      Transcoders. The transcoder is selected depending on the system
      environmental conditions, but there is no particular reason
      to use a transcoder instead of a direct stream, if this vm and
      functions using the standard streams abstractions held in the VM
      are willingt to operate on binary streams or on streams with
      different encodings.

      \note the VM uses the default std error stream to feed the
      standard Default Error Handler at its creation. That handler
      has its own copy of a StdErrorStream implementation, and changing
      this object won't affect the default error handler. If the
      error handler output has to be redirected as well, the new
      stream must be directly set. Also, be sure not to give ownership
      of the stream to the default handler, as giving ownership
      would cause the handler AND the VM to try to destroy the
      same pointer.

      \return current incarnation of standard error stream.
   */
   Stream *stdErr() const { return m_stdErr; }

   /** Set stdandard input stream.
      Old standard input stream abstraction is destroyed (vm owns it).
      \see stdIn()
      \param nstream the new stream
   */
   void stdIn( Stream *nstream );

   /** Set stdandard output stream.
      Old standard output stream abstraction is destroyed (vm owns it).
      \see stdOut()
      \param nstream the new stream
   */
   void stdOut( Stream *nstream );

   /** Set stdandard error stream.
      Old standard error stream abstraction is destroyed (vm owns it).
      \note This doesn't changes the stream used by the standard error handler, which,
            by default, is set to a owned copy of this stream.
      \see stdOut()
      \param nstream the new stream
   */
   void stdErr( Stream *nstream ) ;


   bool hasProcessStreams() const { return m_bhasStandardStreams; }

   /** Indicastes if this VM has been given standard streams control or not.
      VM in embedding applications hold generally streams that are not pointing to the
      process streams. Only direct itepreters (e.g. falcon command line) generally
      provides the VM with the process standard streams; normally, the process doesn't
      want the VM to use its standard stream, and even when it does, it doesn't want the
      VM (and the scripts) to be able to close the streams on its behalf.

      This is however desirable when scripts are running in a standalone environment,
      so that they can interact with other piped processes. In that case, both embedding
      applications, but more generally script interpreters as falcon command line, will
      give full stream ownership to the VM, and let them to possibly close the streams
      directly.

      \param b set to true to allow the scripts on this VM to close standard streams.
   */
   void hasProcessStreams( bool b ) { m_bhasStandardStreams = b; }

   /** Error handler implementation.
      Creates an error object and raises it in the current script.
      The script can then intercept the raised error, or let it flow
      through. If going thoruh, this same error will be passed to the
      error handler of this VM.
   */

   virtual void handleError( Error *error );

   /** Return the exit error.
      If the virutal machine exited uncleanly, i.e. because of a raised error,
      this variable contains an instance of the error that caused premature
      termination.

      The variable belongs to the VM and is destroyed with it.
   */
   Error *exitError() const { return m_error; }

   /** True if the VM exited with error.
      If the VM is cleanly terminated, this method returns false.
      Useful to know if the script did what it was supposed to do.
   */
   bool hadError() const { return m_event == eventRisen; }

   /** In case of VM sleeps, return to the main application.
      If a yield request has been sent to the VM, and that
      request would generate a vm sleep, the default behavior is that of
      stopping the process for a given time.

      However, the embedding application may wish to get in control
      during this waits, either to perform some background tasks
      or to accept asynchronous messages during the wait.

      To have the VM to return instead of waiting when some sleep request
      is processed, call this method with true as parameter.

      When the VM returns this way, hadSleepreQuest() returns true,
      lastEvent() is set to sleepRequest and sleepRequestTime
      returns the number of seconds after which the VM should
      be called again.

      \note to re-enable the VM after this wait, just call its run()
      method. Remember to reset the sleep event with resetEvent() before
      relaunching the VM.

      \param r true to have the VM exit the run() loop instaed of waiting.
   */
   void returnOnSleep( bool r ) { m_sleepAsRequests = r; }

   /** Checks the current status of the returnOnSleep( bool ) setting.
      \return true if wait requests causes a VM return.
   */
   bool returnOnSleep() const { return m_sleepAsRequests; }

   /** True if the VM exited because of sleep request.
      The embedding application should sleep for the time specified
      in sleepRequestTime().

      To have the VM to return from the run() loop with this request
      set, instead of waiting on its own, the embedding application
      should call returnOnSleep( bool ) method.

      \return true if vm exited because of a sleep request.
   */
   bool hadSleepRequest() const { return m_event == eventSleep; }

   /** Time for which the embedding application should sleep.
      When the VM exits the run() loop with the hadSleepReuqest()
      method returning true (or lastEvent() == eventSleep ), this
      number indicates the number of seconds the embedding application
      should sleep.

      \return number of seconds to wait (with fractional parts).
   */
   numeric sleepRequestTime() const { return m_yieldTime; }

   /** Return current module string at given ID.
      Retreives the module string which has been given the passed ID,
      and returns it to the caller. If the string is not found, a
      statically allocated empty string is returned instead.
      \param stringId the id for the string in the current module
   */
   const String &moduleString( uint32 stringId ) const;

   /** Reset machine for a clean execution.
      This resets the VM to execute cleanly a ascript, removing all
      the dirty variables, execution context and the rest.
   */
   void reset();

   void resetEvent() { m_event = eventNone; }
   void limitLoops( uint32 l ) { m_opLimit = l; }
   uint32 limitLoops() const { return m_opLimit; }
   bool limitLoopsHit() const { return m_opLimit >= m_opCount; }

   void resetCounters();

   /** Performs a single VM step and return. */
   void step() {
      m_pc = m_pc_next;
      m_bSingleStep = true;
      // stop next loop
      m_opNextCheck = m_opCount + 1;
      run();
   }

   void singleStep( bool ss ) { m_bSingleStep = ss; }
   bool singleStep() const { return m_bSingleStep; }

   /** Compare two items.
      If the first comparand is an object, compare() and eventually equal()
      are searched in the object definition, and then executed.
      If compare() is not found, and equal() is not found or does not return
      true, a normal item comparation is performed.

      This method is used in all the VM basic comparations.
   */
   int compareItems( const Item &first, const Item &second );

   /** Periodic callback.
      This is the periodic callback routine. Subclasses may use this function to get
      called every now and then to i.e. stop the VM asynchronously, or to perform
      debugging.
   */
   virtual void periodicCallback();

   void callbackLoops( uint32 cl ) { m_loopsCallback = cl; }
   uint32 callbackLoops() const { return m_loopsCallback; }

   void gcCheckLoops( uint32 cl ) { m_loopsGC = cl; }
   uint32 gcCheckLoops() const { return m_loopsGC; }

   void contextCheckLoops( uint32 cl ) { m_loopsContext = cl; }
   uint32 contextCheckLoops() const { return m_loopsContext; }

   /** Return the loops that this VM has performed.
      There is no guarantee that the loops performed by the virtual
      machine hadn't overflown the uint32 size, nuless there is a
      physical loop limit set with loopLimit().
      However, this counter can be useful to check different algorithm
      performances.
   */
   uint32 elapsedLoops() const { return m_opCount; }


   /** Push current try position */
   void pushTry( int32 landingPC );

   /** Pop a try position, eventually changing the frame to the handler. */
   void popTry( bool moveTo );

   /** Elects a new context ready for execution.
      This method should be called by embedding applications that have performed
      a sleep operations right after the elapsed sleep time. The VM will elect
      the most suitable context for execution.

      On output, the VM will be ready to run (call the run() function); if no context
      is willing to run yet, there this method will set the eventSleep (hadSleepRequest()
      will return true), and the sleep request time will be propery set.
      \see returnOnSleep()
   */
   void electContext();

   /** Types of return values.
      When a function may return more than simply syccess or failure, this code
      specifies what went wrong in a VM function.
   */
   typedef enum {
      return_ok,
      return_error_string,
      return_error_parse,
      return_error_internal,
      return_error_parse_fmt
   }
   returnCode;

   /** Performs a string expansion.
      \return a returnCode enumeration explaining possible errors in string expansion
   */
   returnCode  expandString( const String &src, String &target );

   void store( Garbageable *data ) { m_memPool->storeForGarbage( data ); }
   void store( GarbageString *str_data ) { m_memPool->storeForGarbage( str_data ); }

   /** Creates a reference to an item.
      The source item is turned into a reference which is passed in the
      target item. If the source is already a reference, the reference
      is just passed in the target.
      \param target the item that will accept reference to source
      \param source the source item to be referenced
   */
   void referenceItem( Item &target, Item &source );

   /** Ensures that the given item is never destroyed by the garbage collector.
      At times, embedding applications takes references to items so that they
      can directly access them later. In example, a script may declare a callback
      method, and then lose reference of that callable object. The embedding
      application will want to have the callable object to stay alive, wether
      the script wants to save it or not.

      One solution may be that of recording the item somewhere in the VM, in
      example in an hidden global array where all the "saved" item are stored,
      but that's quite expensive.

      Instead, the embedding application may just have a local copy of the item
      and call "gcLock" on it. This ensures that the data stay valid until the
      VM is alive even if the item contains deep (garbage sensible) data, and
      the script lose track of it.

      When the data is not needed anymore (i.e. because the callback handler is
      destroyed externally), the embedding application may call gcUnlock() to
      allow VM and Garbabe Collector to reclaim that memory in case it is needed.

      \note This method is currently implemented using the underlying MemPool::lock
         method, and it follows the same rules. Be careful re-locking items twice or
         unlocking non-locked items, or you'll crash.

      \see MemPool::lock
      \see gcUnlock( const Item & )
      \see GarbageItem
      \param itm the item to be locked
      \return a relocable item pointer that can be used to access the deep data.
   */
   GarbageLock *gcLock( const Item &itm ) { return m_memPool->lock( itm ); }

   /** Allows a previously locked item to be reclaimed (later on).
      \param itm the item to be unlocked
      \see gcLock( const Item & )
   */
   void gcUnlock( GarbageLock *itm ) { m_memPool->unlock( itm ); }

   uint32 stackBase() const { return m_stackBase; }

//==========================================================================
//==========================================================================
//==========================================================================

   // Range 1: Parameterless ops
   /** End opcode handler.
      END opcode terminates current coroutine or current virtual machine execution.
   */
   friend void opcodeHandler_END ( register VMachine *vm );
   friend void opcodeHandler_NOP ( register VMachine *vm );
   friend void opcodeHandler_PSHN( register VMachine *vm );
   friend void opcodeHandler_RET ( register VMachine *vm );
   friend void opcodeHandler_RETA( register VMachine *vm );
   friend void opcodeHandler_PTRY( register VMachine *vm );

   // Range 2: one parameter ops;
   friend void opcodeHandler_LNIL( register VMachine *vm );
   friend void opcodeHandler_RETV( register VMachine *vm );
   friend void opcodeHandler_FORK( register VMachine *vm );
   friend void opcodeHandler_BOOL( register VMachine *vm );
   friend void opcodeHandler_GENA( register VMachine *vm );
   friend void opcodeHandler_GEND( register VMachine *vm );
   friend void opcodeHandler_PUSH( register VMachine *vm );
   friend void opcodeHandler_PSHR( register VMachine *vm );
   friend void opcodeHandler_POP ( register VMachine *vm );
   friend void opcodeHandler_JMP ( register VMachine *vm );
   friend void opcodeHandler_INC ( register VMachine *vm );
   friend void opcodeHandler_DEC ( register VMachine *vm );
   friend void opcodeHandler_NEG ( register VMachine *vm );
   friend void opcodeHandler_NOT ( register VMachine *vm );
   friend void opcodeHandler_TRAL( register VMachine *vm );
   friend void opcodeHandler_IPOP( register VMachine *vm );
   friend void opcodeHandler_XPOP( register VMachine *vm );
   friend void opcodeHandler_GEOR( register VMachine *vm );
   friend void opcodeHandler_TRY ( register VMachine *vm );
   friend void opcodeHandler_JTRY( register VMachine *vm );
   friend void opcodeHandler_RIS ( register VMachine *vm );
   friend void opcodeHandler_BNOT( register VMachine *vm );
   friend void opcodeHandler_NOTS( register VMachine *vm );
   friend void opcodeHandler_PEEK( register VMachine *vm );

   // Range3: Double parameter ops;
   friend void opcodeHandler_LD  ( register VMachine *vm );
   friend void opcodeHandler_LDRF( register VMachine *vm );
   friend void opcodeHandler_ADD ( register VMachine *vm );
   friend void opcodeHandler_SUB ( register VMachine *vm );
   friend void opcodeHandler_MUL ( register VMachine *vm );
   friend void opcodeHandler_DIV ( register VMachine *vm );
   friend void opcodeHandler_MOD ( register VMachine *vm );
   friend void opcodeHandler_POW ( register VMachine *vm );
   friend void opcodeHandler_ADDS( register VMachine *vm );
   friend void opcodeHandler_SUBS( register VMachine *vm );
   friend void opcodeHandler_MULS( register VMachine *vm );
   friend void opcodeHandler_DIVS( register VMachine *vm );
   friend void opcodeHandler_MODS( register VMachine *vm );
   friend void opcodeHandler_BAND( register VMachine *vm );
   friend void opcodeHandler_BOR ( register VMachine *vm );
   friend void opcodeHandler_BXOR( register VMachine *vm );
   friend void opcodeHandler_ANDS( register VMachine *vm );
   friend void opcodeHandler_ORS ( register VMachine *vm );
   friend void opcodeHandler_XORS( register VMachine *vm );
   friend void opcodeHandler_GENR( register VMachine *vm );
   friend void opcodeHandler_EQ  ( register VMachine *vm );
   friend void opcodeHandler_NEQ ( register VMachine *vm );
   friend void opcodeHandler_GT  ( register VMachine *vm );
   friend void opcodeHandler_GE  ( register VMachine *vm );
   friend void opcodeHandler_LT  ( register VMachine *vm );
   friend void opcodeHandler_LE  ( register VMachine *vm );
   friend void opcodeHandler_IFT ( register VMachine *vm );
   friend void opcodeHandler_IFF ( register VMachine *vm );
   friend void opcodeHandler_CALL( register VMachine *vm );
   friend void opcodeHandler_INST( register VMachine *vm );
   friend void opcodeHandler_ONCE( register VMachine *vm );
   friend void opcodeHandler_LDV ( register VMachine *vm );
   friend void opcodeHandler_LDP ( register VMachine *vm );
   friend void opcodeHandler_TRAN( register VMachine *vm );
   friend void opcodeHandler_UNPK( register VMachine *vm );
   friend void opcodeHandler_SWCH( register VMachine *vm );
   friend void opcodeHandler_HAS ( register VMachine *vm );
   friend void opcodeHandler_HASN( register VMachine *vm );
   friend void opcodeHandler_GIVE( register VMachine *vm );
   friend void opcodeHandler_GIVN( register VMachine *vm );
   friend void opcodeHandler_IN  ( register VMachine *vm );
   friend void opcodeHandler_NOIN( register VMachine *vm );
   friend void opcodeHandler_PROV( register VMachine *vm );
   friend void opcodeHandler_SSTA( register VMachine *vm );
   friend void opcodeHandler_SVAL( register VMachine *vm );
   friend void opcodeHandler_STPS( register VMachine *vm );
   friend void opcodeHandler_STVS( register VMachine *vm );
   friend void opcodeHandler_LSTA( register VMachine *vm );
   friend void opcodeHandler_LVAL( register VMachine *vm );
   friend void opcodeHandler_AND ( register VMachine *vm );
   friend void opcodeHandler_OR  ( register VMachine *vm );
   friend void opcodeHandler_PASS( register VMachine *vm );
   friend void opcodeHandler_PSIN( register VMachine *vm );

   // Range 4: ternary opcodes;
   friend void opcodeHandler_STP ( register VMachine *vm );
   friend void opcodeHandler_STV ( register VMachine *vm );
   friend void opcodeHandler_LDVT( register VMachine *vm );
   friend void opcodeHandler_LDPT( register VMachine *vm );
   friend void opcodeHandler_STPR( register VMachine *vm );
   friend void opcodeHandler_STVR( register VMachine *vm );
   friend void opcodeHandler_TRAV( register VMachine *vm );

   friend void opcodeHandler_SJMP( register VMachine *vm );
   friend void opcodeHandler_FORI( register VMachine *vm );
   friend void opcodeHandler_FORN( register VMachine *vm );

   friend void opcodeHandler_SHL( register VMachine *vm );
   friend void opcodeHandler_SHR( register VMachine *vm );
   friend void opcodeHandler_SHLS( register VMachine *vm );
   friend void opcodeHandler_SHRS( register VMachine *vm );
   friend void opcodeHandler_LDVR( register VMachine *vm );
   friend void opcodeHandler_LDPR( register VMachine *vm );
   friend void opcodeHandler_POWS( register VMachine *vm );
   friend void opcodeHandler_LSB( register VMachine *vm );
   friend void opcodeHandler_UNPS( register VMachine *vm );
   friend void opcodeHandler_SELE( register VMachine *vm );
   friend void opcodeHandler_INDI( register VMachine *vm );
   friend void opcodeHandler_STEX( register VMachine *vm );
   friend void opcodeHandler_TRAC( register VMachine *vm );
   friend void opcodeHandler_WRT( register VMachine *vm );
};

}

#endif

/* end of flc_vm.h */

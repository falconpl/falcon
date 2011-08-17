/*
   FALCON - The Falcon Programming Language.
   FILE: vm.h

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Dom, 08 Aug 2004 01:22:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_VM_H
#define FLC_VM_H

#include <falcon/symtab.h>
#include <falcon/symlist.h>
#include <falcon/item.h>
#include <falcon/stackframe.h>
#include <falcon/module.h>
#include <falcon/common.h>
#include <falcon/error.h>
#include <falcon/runtime.h>
#include <falcon/vmmaps.h>
#include <falcon/genericlist.h>
#include <falcon/carray.h>
#include <falcon/string.h>
#include <falcon/coreobject.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/genericmap.h>
#include <falcon/genericlist.h>
#include <falcon/fassert.h>
#include <falcon/vm_sys.h>
#include <falcon/coreslot.h>
#include <falcon/baton.h>
#include <falcon/livemodule.h>
#include <falcon/vmcontext.h>
#include <falcon/mersennetwister.h>

#define FALCON_VM_DFAULT_CHECK_LOOPS 5000

namespace Falcon {

class Runtime;
class VMachine;
class PropertyTable;
class AttribHandler;
class MemPool;
class VMMessage;
class GarbageLock;


typedef void (*tOpcodeHandler)( register VMachine *);

void ContextList_deletor( void * );

class FALCON_DYN_CLASS ContextList: public List
{
   friend void ContextList_deletor( void * );

public:
   ContextList()
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
void opcodeHandler_LDAS( register VMachine *vm );
void opcodeHandler_SWCH( register VMachine *vm );
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
void opcodeHandler_INCP( register VMachine *vm );
void opcodeHandler_DECP( register VMachine *vm );

void opcodeHandler_SHL( register VMachine *vm );
void opcodeHandler_SHR( register VMachine *vm );
void opcodeHandler_SHLS( register VMachine *vm );
void opcodeHandler_SHRS( register VMachine *vm );
void opcodeHandler_CLOS( register VMachine *vm );
void opcodeHandler_PSHL( register VMachine *vm );

void opcodeHandler_POWS( register VMachine *vm );
void opcodeHandler_LSB( register VMachine *vm );
void opcodeHandler_SELE( register VMachine *vm );
void opcodeHandler_INDI( register VMachine *vm );
void opcodeHandler_STEX( register VMachine *vm );
void opcodeHandler_TRAC( register VMachine *vm );
void opcodeHandler_WRT( register VMachine *vm );
void opcodeHandler_STO( register VMachine *vm );
void opcodeHandler_FORB( register VMachine *vm );
void opcodeHandler_EVAL( register VMachine *vm );
void opcodeHandler_OOB( register VMachine *vm );
void opcodeHandler_TRDN( register VMachine *vm );
void opcodeHandler_EXEQ( register VMachine *vm );


class VMachine;

class FALCON_DYN_CLASS VMBaton: public Baton
{
   VMachine *m_owner;

public:
   VMBaton( VMachine* owner ):
      Baton( true ),
      m_owner( owner )
   {}
   virtual ~VMBaton() {}

   virtual void release();
   virtual void onBlockedAcquire();
   void releaseNotIdle();
};

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
class FALCON_DYN_CLASS VMachine: public BaseAlloc
{
   friend class MemPool;

public:

   /** This valuess indicates that the VM is in external execution mode.
      With this value in m_pc, the VM will execute the symbol found
      in m_symbol.
   */
   enum {
      i_pc_call_request = 0xFFFFFFFF-sizeof(int32)*4,
      i_pc_redo_request =0xFFFFFFFF-sizeof(int32)*4,
      i_pc_call_external_ctor = 0xFFFFFFFF-sizeof(int32)*3,
      i_pc_call_external_ctor_return = 0xFFFFFFFF-sizeof(int32)*2,
		i_pc_call_external = 0xFFFFFFFF-sizeof(int32),
		i_pc_call_external_return = 0xFFFFFFFF
   };

protected:
   /** Structure used to save the vardefs during class link */
   class VarDefMod: public BaseAlloc
   {
   public:
      VarDef *vd;
      LiveModule *lmod;
   };


   mutable Mutex m_mtx;

   /** Mutex guarding the slot structure. */
   mutable Mutex m_slot_mtx;

    /** Main module.
      This is the last linked module, that should also be the module where to search
      for start symbols; by default, launch() searches symbol here,
      and if not found, they search start symbols in globally exported symbol tables.
   */
   LiveModule *m_mainModule;


   /** Space for immediate operands. */
   Item m_imm[4];

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;
   bool m_bhasStandardStreams;

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

   /** Generation at which the garbage data in this VM is marked. */
   uint32 m_generation;

   /**  Map of live modules.
      Todo: better docs.
   */
   LiveModuleMap m_liveModules;

   /** This value indicate that there isn't any active try handler in the stack */
   enum {
      i_noTryFrame = 0xFFFFFFFF
   };

   /** Context currently being executed. */
   VMContext *m_currentContext;

   /** Ready to run contexts. */
   ContextList m_contexts;
   /** Contexts willing to sleep for a while */
   ContextList m_sleepingContexts;
   /** Wether or not to allow a VM hostile takeover of the current context. */
   bool m_allowYield;

   /** Execute at link time? */
   bool m_launchAtLink;

   /** Opcode handler function calls. */
   tOpcodeHandler *m_opHandlers;

   /** Map of global symbols (and the item they are connected to).
      Each item of the map contains a Symbol * and an ID that allows to
   */
   SymModuleMap m_globalSyms;

   /** Map of well knwon symbols (and the item they are connected to).
      Each item of the map contains a Symbol * and an ID that allows to
   */
   SymModuleMap m_wellKnownSyms;

   /** Puts the given context at sleep.
      This function inserts the given context in the sleep queue, inserting it in
      schedule order.

      Contexts waiting forever are put at bottom; contexts not wishing to sleep
      are put at top.
   */
   void putAtSleep( VMContext *ctx );

   /** Resort this context changing its position in the sleep list.

      Actually, this function works as putAtSleep(), but it
      continues to scan the sleeping context sleep till it finds
      the previous position of the context to remove it.
   */
   void reschedule( VMContext *ctx );

   /** Service recursive function called by LinkClass to create a class. */
   bool linkSubClass( LiveModule *mod , const Symbol *clssym, Map &props, Map &states, ObjectFactory *factory );

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

   /** User available pointer.
      \see userData( void * );
   */
   void *m_userData;

   /** System specific data.
      \see systemData();
   */
   Sys::SystemData m_systemData;

   CoreClass **m_metaClasses;

   Map m_slots;

   VMachine *m_nextVM;
   VMachine *m_prevVM;

   VMachine *m_idleNext;
   VMachine *m_idlePrev;

   /** If true, the VM allows to be blocked periodically by the GC in case of need.

      If false, the VM is left alone, and in case it becomes the VM having taken a GC
      scan for the last, its GC data gets promoted only if the GC enters the active
      state.
   */
   bool m_bGcEnabled;

   /** Main synchronization baton. */
   VMBaton m_baton;

   Mutex m_mtx_mesasges;
   VMMessage* m_msg_head;
   VMMessage* m_msg_tail;

   /** Event set by the VM to ask for priority GC.
      This is used by the performGC() function to inform the GC loop about the priority
      of this VM.
   */
   bool m_bPirorityGC;

   /** Event set by the GC to confirm the forced GC loop is over. */
   Event m_eGCPerformed;

   /** True when we want to wait for collection before being notified in priority scans. */
   bool m_bWaitForCollect;

   /** Mutex for locked items ring.
    *  -- unused; kept for binary compatibilty
    * */
   Mutex m_mtx_lockitem;

   /** Locked and unreclaimable items are stored in this ring.
    * -- unused; kept for binary compatibilty
    * */
   GarbageLock *m_lockRoot;

   /** Reference count.
      Usually, shouldn't be very high: the caller program, the GC and possibly
      some extra method in embedding applications.
   */
   volatile int m_refcount;

   /** True if current frame should break */
   bool m_break;

   /** Finalization hook for MT system. */
   void (*m_onFinalize)(VMachine *vm);

   /** filtered load path */
   String m_appSearchPath;

   /** random number generator */
   MTRand _mtrand;

   //=============================================================
   // Private functions
   //

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
      register int32 ret = *reinterpret_cast<int32 *>(
         m_currentContext->code() + m_currentContext->pc_next()  );
      m_currentContext->pc_next() += sizeof( int32 );
      return ret;
   }

   /** Returns the next NTD64 parameter, advancing the pointer to the next instruction */
   int64 getNextNTD64()
   {
      register int64 ret = loadInt64(
         m_currentContext->code() + m_currentContext->pc_next() );
      m_currentContext->pc_next() += sizeof( int64 );
      return ret;
   }

   /** Gets the nth parameter of an opcode. */
   Item *getOpcodeParam( register uint32 bc_pos );


   /** Perform an item raise.

      This implements the "raise" keyword or "RIS" opcode, that is,
      takes an item and sends it to the relevant catch in the call
      hierarcy.

      If the item can't be caught by the current context, then it is
      eventually encapsulated in an error and thrown to the applicaiton
      with a C++ throw new Error*. If the item contains an instance of
      a Falcon Error class, the inner core Error* is taken and that is
      raised instead.

      After this call, if the item or error could be caught by the script,
      the context is prepared to run the error handler at the very next
      VM loop.

      @param value The item to be raised, possibly but not necessarily
      derived from a Falcon level Error class.
   */
   void handleRaisedItem( Item& value );

   /** Decides what to do with an error incoming in the main loop.

      Usually, this get either rethrown to the application or
      handled as an item down to the script.

      Stack is eventually unrolled till the item handler is found.
   */
   void handleRaisedError( Error* err );

   /** Performs periodic checks on the virtual machine. */
   void periodicChecks();

   /** Creates a new stack frame in the current context
      \param paramCount number of parameters in the stack
      \param frameEndFunc Callback function to be executed at frame end
   */
   void createFrame( uint32 paramCount, ext_func_frame_t frameEndFunc = 0 )
   {
      m_currentContext->prepareFrame( paramCount, frameEndFunc );
   }

   /** Sets the currently running VM.
      The Currently running VM is the VM currently in control
      of garbage collecting and item creation.

      There can be only one current VM per thread; the value is
      stored in a thread specific variable.

      Accessing it can be relatively heavy, so it is highly advised
      not tu use it except when absolutely necessary to know
      the current vm and the value is currently unknown.

      \note Embedding applications are advised not to "corss VM", that is,
      not to nest call into different VMs in the same thread
   */
   void setCurrent() const;

   /** Process all the pending messages.
    * Pending mesasges are sent to the processMessage() method, which
    * creates a coroutine context ready to be fired as soon as the VM
    * gets back in action.
    *
    * The processing order is not granted.
    */
   void processPendingMessages();

   /** Processes an incoming message.
      This searches for the slot requierd by the message;
      if it is found, the message is broadcast to the slot in a newly created coroutine,
      otherwise the onMessageComplete is immedately called.
   void terminateCurrentContext();
      The message will be immediately destroyed if it can't be broadcast.

      \param msg The message to be processed.
   */
   void processMessage( VMMessage* msg );

   virtual bool linkDefinedSymbol( const Symbol *sym, LiveModule *lmod );
   virtual bool linkUndefinedSymbol( const Symbol *sym, LiveModule *lmod );
   virtual bool completeModLink( LiveModule *lmod );
   virtual LiveModule *prelink( Module *mod, bool bIsMain, bool bPrivate );

   void raiseHardError( int code, const String &expl, int32 line );

   /** Gets the next item in a trav loop */
   Item* getNextTravVar();

   /** Prepare a coroutine context.

     The context is immediately swapped, so any operation performed by
     the caller is done on the new context. This allow to call an item
     right after coroutine creation.

     \param paramCount Number of parameters to be passed in the coroutine stack.
   */
   VMContext* coPrepare( int32 paramCount );

   /** Destroys the virtual machine.
      Protected as it can't be called directly.
   */
   virtual ~VMachine();

public:
   /** Returns the currently running VM.

      The Currently running VM is the VM currently in control
      of garbage collecting and item creation.

      There can be only one current VM per thread; the value is
      stored in a thread specific variable.

      Accessing it can be relatively heavy, so it is highly advised
      not tu use it except when absolutely necessary to know
      the current vm and the value is currently unknown.

      \note Embedding applications are advised not to "corss VM", that is,
      not to nest call into different VMs in the same thread
   */
   static VMachine *getCurrent();

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


   /** Initialize the virutal machine.
      Setups some configurable items that are used by the machine.

      They are namely:

      - m_stdIn: standard input. The default is to use an instance returned by
        stdInputStream(), which interfaces a localized text version of
        process stdin.
      - m_stdOut: standard output. The default is to use an instance returned by
        stdOutputStream(), which interfaces a localized text version of
        process stdout.
      - m_stdErr: standard error. The default is to use an instance returned by
        stdErrorStream(), which interfaces a localized text version of
        process stderr.   void terminateCurrentContext();

      The subclass should set up its own items, if the default ones are not
      suitable, and then call the base class init(). The base init will
      not overwrite those items if they are valorized, but it may configure
      them to set up them properly, so it should always be called even if
      all the items are set by the subclass.

      \see VMachine( bool )
   */
   void init();

   /** Links a set of modules stored in a runtime.
      The modules that have been loaded and pre-linked by the runtime
      are correctly inserted in the VM. The topmost module in the
      Runtime is set as main module.

      After the link step, the runtime is not needed anymore and can
      be destroyed; the modules are safely referenced in the VM.

      In case of link error, the error handler of the VM is signaled and the
      function returns false, otherwise it returns true. If the VM hasn't an
      error handler set, nothing is signaled, but the error is still available
      after return through exitError() method.

      \note The main module is the module that is preferentially searched
            for start symbol(s) by prepare() function.
      \param rt the runtime to be linked
      \return The topmost module in the runtime turned into a livemoude, or zero on failure.
   */
   virtual LiveModule* link( Runtime *rt );

   /** Links a single module.
      The last-linked module is usually set as the main module, but it is possible
      to link a non-main module.

      After linking, the caller may release the reference if the module is needed
      only in this VM; the VM keeps a reference to the module.

      The VM holds a reference to the module also in case of errors: the module
      may be still needed for error reports or debug. To destroy definitely the
      module, the VM must be destroyed or the module must be unlinked.

      In case of link error, the error handler of the VM is signaled and the
      function returns false, otherwise it returns true. If the VM hasn't an
      error handler set, nothing is signaled, but the error is still available
      after return through exitError() method.

      \note The main module is the module that is preferentially searched
            for start symbol(s) by prepare() function.
      \param rt the runtime to be linked
      \param isMainModule false to prevent this module to be chosen as startup module.
      \param bPrivate false to allow this module to export the exported symbols, true
               to make the module private (and prevent exports).
      \return 0 time error, the internally built LiveModule instance on success.
   */
   virtual LiveModule *link( Module *module, bool isMainModule=true, bool bPrivate = false );

   /** Links a single symbol on an already existing live module.
      This won't actually link instances and classes which must be post-processed.
      \param sym The symbol to be linked.
      \param lmod The live module where the symbol must go.
      \return false if the link fails, true on success.
   */
   virtual bool linkSymbol( const Symbol *sym, LiveModule *lmod );


   /** Try to link a dynamic symbol.

      This method asks the modules that can provide dynamic symbols if they
      are interested to serve us with a symbol for the given name. If so,
      the symbol is linked, and a complete reference to its position (SymModule) is
      returned.

      The symbol needs not to be global to be exported this way. The providers may
      return private symbols that will be used just in this occasion and won't enter
      the global symbol map.

      The method may raise any error that linkSymbolComplete may raise. The same
      cares used for LinkSymbolComplete   void terminateCurrentContext(); should be used.

      The method is virtual, so subclasses are able to create symbols dynamically
      by providing them directly. However, subclasses creating directly symbols
      must still create them inside a FlexyModule and use linkCompleteSymbol
      to bless the symbol in the VM.

      It is advisable to call the base class version of the method on subclass
      default.

      \param name The symbol to be searched for.
      \param symdata Coordinates of the linked symbol, on success.
      \return true on success, false if the symbol is not found or if it was found
         but couldn't be linked.
   */
   virtual bool linkSymbolDynamic( const String &name, SymModule &symdata );

   /** Links a class symbol.

      Class symbols must be specially post-processed after all the other symbols
      (including themselves) are pre-linked, so that cross references can be resolved.

      This method preforms the creation of a CoreClass that will reside in the live
      module passed as parameter.

      In debug, an assert checks that the incoming symbol is a class. If dynamically
      called, the caller should check that \b sym is a real class symbol.

      \param sym The class symbol to be linked.
      \param lmod The live module where the symbol must go.
      \return false if the link fails, true on success.
   */
   virtual bool linkClassSymbol( const Symbol *sym, LiveModule *livemod );

   /** Links a class instance.

      Class instances are what is commonly called "Singleton objects", declared through
      the "object" Falcon keyword.

      Class instances must be specially post-processed after all the other symbols
      and class declarations are pre-linked, so that cross references can be resolved.

      This method preforms the creation of a CoreClass that will reside in the live
      module passed as parameter.

      In debug, an assert checks that the incoming symbol is a class. If dynamically
      called, the caller should check that \b sym is a real class symbol.

      \param sym The instance symbol to be linked.
      \param lmod The live module where the symbol must go.
      \return false if the link fails, true on success.
   */
   virtual bool linkInstanceSymbol( const Symbol *sym, LiveModule *livemod );

   /** Constructs an instance calling its _init method if necessary.

      Instance construction must be performed after all the items are pre-linked,
      all the classes are formed and all the instances are set-up; in this way,
      it is possible for object _init method to refer to other objects, even if
      not yet initialized, as their structure is complete and coherent in the VM.

      The VM may call either its own VM bytecode or external functions; calls are
      performed in non-atomic mode. This means that this method should not be called
      from functions executed by the VM, normally. If this is necessary, then
      the call of this method must be surrounded by setting and resetting the
      atomic mode.

      \note Running in atomic mode doesn't mean "locking the VM"; the atomic mode is just a mode
      in which the VM runs without switching coroutines, accepting sleep requests or
      fulfilling suspension requests. Trying to suspend or prehempt the VM in atomic
      mode raises an unblockable exception, terminating the current script (and leaving it
      in a non-resumable status).

      \note This method won't apply the init state, if existing.

      \param sym The class instance to be initialized.
      \param lmod The live module where the symbol must go.
      \throw Error if the instance cannot be created.
   */
   virtual void initializeInstance( const Symbol *sym, LiveModule *livemod );

   /** Links a symbol eventually performing class and instances initializations.

      This method should be called when incrementally adding symbol once is knonw
      that the classes and objects they refer to are (or should) be already linked
      in the VM, in already existing live modules.

      This is the case of dynamic symbol creation in flexy modules, where
      the running scripts can create new symbols or invoke module actions that will
      create symbols; this is perfectly legal, but in that case the created instance
      will need to have everything prepared; cross references won't be resolved.

      This also means that the caller willing to link completely a new instance symbol
      must first link it's referenced class.

      This method should run in atomic mode (see initializeInstance() ).
   */
   virtual bool linkCompleteSymbol( const Symbol *sym, LiveModule *livemod );

   /** Links a symbol eventually performing class and instances initializations.

      This method resoves the module name into its liveMod before performing
      complete linking. This allows external elements (i.e. FALCON_FUNC methods)
      to create new symbols on the fly, declaring just into which module they
      should be created.

      If the target module is not found the method returns false, otherwise it calls directly
      linkCompleteSymbol( Symbol *, LiveModule * ) with the appropriate instance
      of the LiveModule.

      This method should run in atomic mode (see initializeInstance() ).

      \param sym Symbol created dynamically.
      \param moduleName name of the module that has created it and wants it to be
                        inserted in the VM.
   */
   virtual bool linkCompleteSymbol( Symbol *sym, const String &moduleName );

   /** Returns the main module, if it exists.
      Returns an instance of the LiveModule class, that is the local representation
      of a module linked in the VM, that holds informations about the "main" module
      in this virtual machine.
   */
   LiveModule *mainModule() const { return m_mainModule; }

   /** Unlinks all the modules in the runtime.

      The unlinked module(s) become unavailable, and all the callable items referencing
      symbols in the module become uncallable. Exported global variables survive unlinking,
      and their value can still be inspected, modified and discarded after they
      have been unlinked.

      It is not possible to unlink a module which is currently being run (that is,
      which is the module holding the currently executed symbol).

      It is possible to unlink the main module, but a new main module won't be
      automatically elected; unless the start symbol is exported by other modules,
      prepare() will fail if a new main module is not linked in the VM.

      The function may return false either if one of the module in the runtime
      is not present in the VM or if one of them is the "current module". However,
      some of the modules may have got unlinked int he meanwhile, and unlinking
      also dereferences them.

      \param rt the runtime with all the modules to be unlinked
      \return true on success, false on error.
   */
   virtual bool unlink( const Runtime *rt );

   /** Unlinks a module.

      The unlinked module become unavailable, and all the callable items referencing
      symbols in the module become uncallable. Exported global variables survive unlinking,
      and their value can still be inspected, modified and discarded after they
      have been unlinked.

      It is not possible to unlink a module which is currently being run (that is,
      which is the module holding the currently executed symbol).

      It is possible to unlink the main module, but a new main module won't be
      automatically elected; unless the start symbol is exported by other modules,
      prepare() will fail if a new main module is not linked in the VM.

      The function may return false either if the module
      is not present in the VM or if it is the "current module".

      \param module the module to be unlinked.
      \return true on success, false on error.
   */
   virtual bool unlink( const Module *module );


   /** Creates a new class live item.
      This function recursively resolves inheritences and constructor of
      classes to generate a "CoreClass", which is a template ready
      to create new instances.

      \param lmod the live module (module + live data) where this class is generated
      \param clsym the symbol where the class is defined.
   */
   virtual CoreClass *linkClass( LiveModule *lmod, const Symbol *clsym );


   /** Launches the "__main__" symbol.
      This is a proxy call to launch( "__main__" );
   */
   void launch() { launch( "__main__" ); }

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
      found an error is raised and false is returned.

      The routine returns true if execution is successful; the VM may return because the routine terminates,
      because of an explicit END opcode and because execution limit (single step or limited step count) is
      matched. Also, uncaught exceptions or explicit requests for termination may cause the VM to return.

      It is possible to know if the execution is complete by looking at the last VM event with the
      lastEvent() method: if none or quit, then the execution is just terminated, else an event is set
      (i.e. tSuspend).

      \param startSym the name of a routine to be executed.
      \param paramCount Number of parameters that have been pushed in the stack as parameters.
      \throw CodeError* if the symbol to be launched couldn't be found.
   */
   void launch( const String &startSym, uint32 paramCount = 0 );


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
      global of the runtime eaxtly as for launch(). This is useful, for example, to provide
      callback entry points in scripts that must maintain their execution status between
      calls. In this way, each run maintain the previous status but grants execution of
      different entry points; also this is quite faster than having to enter the realization
      phase.

      An error will be risen if launch has not been previously called and the routine name
      is not provided.
   */
   virtual void run();

   /** Become target of OS signals. */
   bool becomeSignalTarget()
   {
      return m_systemData.becomeSignalTarget();
   }


   /** Fills an error traceback with the current VM traceback. */
   void fillErrorTraceback( Error &error ) { currentContext()->fillErrorTraceback( error ); }

   /** Returns a single step traceback in the current context. */
   bool getTraceStep( uint32 level, const Symbol* &sym, uint32& line, uint32 &pc )
   {
      return m_currentContext->getTraceStep( level, sym, line, pc );
   }

   /** Get the caller of the current symbol.

      If the caller cannot be found (i.e. because the current symbol is
      called directly by the embedding program) the method returns false.
      \param sym on success, will hold a pointer to the symbol that called the current symbol.
      \param module on success, will hold a pointer to the module where the caller symbol resides.
      \return true on success, false on failure.
   */
   bool getCaller( const Symbol *&sym, const Module *&module);

   /** Get the item that called the current symbol.

      If the caller cannot be found (i.e. because the current symbol is
      called directly by the embedding program) the method returns false.
      \param item on success, will hold the item (eventually the method) that originated the call.
      \param level previous callers desired (0 is the first caller).
      \return true on success, false on failure.
   */
   bool getCallerItem( Item &caller, uint32 level=0 );

   /** Fills an error with current VM execution context and traceback.
   */
   void fillErrorContext( Error *err, bool filltb = true );

   /** Returns the current stack as a reference. */
   ItemArray &stack() { return m_currentContext->stack(); }

   /** Returns the current stack as a reference (const version). */
   const ItemArray &stack() const { return m_currentContext->stack(); }

   /** Returns a reference to the nth item in the current stack. */
   Item &stackItem( uint32 pos ) { return stack()[ pos ]; }

   /** Returns a reference to the nth item in the current stack (const version). */
   const Item &stackItem( uint32 pos ) const { return stack()[pos]; }

   /** Returns the current module global variables vector. */
   ItemArray &currentGlobals() { return m_currentContext->globals(); }

   /** Returns the current module global variables vector (const version). */
   const ItemArray &currentGlobals() const { return m_currentContext->globals(); }

   /** Returns a reference to the nth item in the current module global variables vector. */
   Item &moduleItem( uint32 pos ) { return currentGlobals()[ pos ]; }

   /** Returns a reference to the nth item in the current module global variables vector (const version). */
   const Item &moduleItem( uint32 pos ) const { return currentGlobals()[ pos ]; }

   /** Returns the module in which the execution is currently taking place. */
   const Module *currentModule() const { return m_currentContext->lmodule()->module(); }

   /** Returns the module in which the execution is currently taking place. */
   LiveModule *currentLiveModule() const { return m_currentContext->lmodule(); }

   /** Find a linked module with a given name.
      Returns a pointer to the linked live module if the name exists, or 0 if the named module
      doesn't exist.
   */
   LiveModule *findModule( const String &name );

   /** Return the map of live modules ordered by name.
      \return the list of live (linked) modules.
   */
   LiveModuleMap &liveModules() { return m_liveModules; }

   /** Return the map of live modules ordered by name (const version).
      \return the list of live (linked) modules.
   */
   const LiveModuleMap &liveModules() const { return m_liveModules; }

   /** Returns the parameter count for the current function.
      \note Calling this when the VM is not engaged in executing a function will crash.
      \return parameter count for the current function.
   */
   int32 paramCount() const {
      return currentContext()->paramCount();
   }

   /** Returns the nth paramter passed to the VM.
      Const version of param(uint32).
   */
   const Item *param( uint32 itemId ) const
   {
      return currentContext()->param( itemId );
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

      \note Fetched item pointers are valid while the stack doesn't change.
            Pushes, addLocal(), item calls and VM operations may alter the
            stack. Using this method again after such operations allows to
            get a valid pointer to the desired item. Items extracted with
            this method can be also saved locally in an Item instance, at
            the cost of a a flat item copy (a few bytes).

      \param itemId the number of the parameter accessed, 0 based.
      \return a valid pointer to the (dereferenced) parameter or 0 if itemId is invalid.
      \see isParamByRef
   */
   Item *param( uint32 itemId )
   {
      return currentContext()->param( itemId );
   }

   /** Returns the nth pre-paramter passed to the VM.
      Pre-parameters can be used to pass items to external functions.
      They are numbered 0...n in reverse push order, and start at the first
      push before the first real parameter.

      For example;

      \code
         Item *p0, *p1, *callable;
         ...
         vm->pushParameter( (int64) 0 );   // pre-parameter 1
         vm->pushParameter( vm->self() );  // pre-parameter 0
         vm->pushParameter( *p0 );
         vm->pushParameter( *p1 );

         vm->callFrame( *callable, 2 );    // 2 parameters
      \endcode
   */
   Item *preParam( uint32 itemId )
   {
      return currentContext()->preParam( itemId );
   }

   /** Const version of preParam */
   const Item *preParam( uint32 itemId ) const
   {
      return currentContext()->preParam( itemId );
   }

   /** Returns true if the nth element of the current function has been passed by reference.
      \param itemId the number of the parameter accessed, 0 based.
      \return true if the parameter exists and has been passed by reference, false otherwise
   */
   bool isParamByRef( uint32 itemId ) const
   {
      return currentContext()->isParamByRef( itemId );
   }

   /** Returns the nth local item.
      The first variable in the local context is numbered 0.
      \note Fetched item pointers are valid while the stack doesn't change.
            Pushes, addLocal(), item calls and VM operations may alter the
            stack. Using this method again after such operations allows to
            get a valid pointer to the desired item again. Items extracted with
            this method can be also saved locally in an Item instance, at
            the cost of a a flat item copy (a few bytes).
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   const Item *local( uint32 itemId ) const
   {
      return m_currentContext->local( itemId );
   }

   /** Returns the nth local item.
      This is just the non-const version.
      The first variable in the local context is numbered 0.
      \param itemId the number of the local item accessed.
      \return a valid pointer to the (dereferenced) local variable or 0 if itemId is invalid.
   */
   Item *local( uint32 itemId )
   {
      return m_currentContext->local( itemId );
   }

   const Item &regA() const { return m_currentContext->regA(); }
   Item &regA() { return m_currentContext->regA(); }
   const Item &regB() const { return m_currentContext->regB(); }
   Item &regB() { return m_currentContext->regB(); }
   const Item &regBind() const { return m_currentContext->regBind(); }
   Item &regBind() { return m_currentContext->regBind(); }
   const Item &regBindP() const { return m_currentContext->regBindP(); }
   Item &regBindP() { return m_currentContext->regBindP(); }

   const Item &self() const { return m_currentContext->self(); }
   Item &self() { return m_currentContext->self(); }

   /** Latch item.
      Generated on load property/vector instructions, it stores the accessed object.
   */
   const Item &latch() const { return m_currentContext->latch(); }
   /** Latch item.
      Generated on load property/vector instructions, it stores the accessed object.
   */
   Item &latch() { return m_currentContext->latch(); }

   /** Latcher item.
      Generated on load property/vector instructions, it stores the accessor item.
   */
   const Item &latcher() const { return m_currentContext->latcher(); }
   /** Latcher item.
      Generated on load property/vector instructions, it stores the accessor item.
   */
   Item &latcher() { return m_currentContext->latcher(); }


   void retval( bool val ) {
       m_currentContext->regA().setBoolean( val );
   }

   void retval( int32 val ) {
       m_currentContext->regA().setInteger( (int64) val );
   }

   void retval( int64 val ) {
       m_currentContext->regA().setInteger( val );
   }

   void retval( numeric val ) {
       m_currentContext->regA().setNumeric( val );
   }

   void retval( const Item &val ) {
       m_currentContext->regA() = val;
   }


   /** Returns a non garbageable string. */
   void retval( String *cs )
   {
      m_currentContext->regA().setString(cs);
   }


   /** Returns a garbageable string.

      \note to put a nongarbage string in the VM use regA() accessor, but you must know what you're doing.
   */
   void retval( CoreString *cs )
   {
      m_currentContext->regA().setString(cs);
   }

   void retval( CoreArray *ca ) {
      m_currentContext->regA().setArray(ca);
   }

   void retval( MemBuf* mb ) {
      m_currentContext->regA().setMemBuf(mb);
   }

   void retval( CoreDict *cd ) {
      m_currentContext->regA().setDict( cd );
   }

   void retval( CoreObject *co ) {
      m_currentContext->regA().setObject(co);
   }

   void retval( CoreClass *cc ) {
      m_currentContext->regA().setClass(cc);
   }

   void retval( const String &cs ) {
      CoreString *cs1 = new CoreString( cs );
      cs1->bufferize();
      m_currentContext->regA().setString( cs1 );
   }

   void retnil() { m_currentContext->regA().setNil();}

   const Symbol *currentSymbol() const { return m_currentContext->symbol(); }
   uint32 programCounter() const { return m_currentContext->pc(); }

   const SymModule *findGlobalSymbol( const String &str ) const;

   /** Make this context to sleep and elect a new one.

      If no other context can be elected, the VM may issue an
      onIdleTime() and eventually sleep a bit.
   */
   void yield( numeric seconds );

   void rotateContext();
   void terminateCurrentContext();

   /** Returns a well known item.
      A well known item is an item that does not phiscally resides in any module, and is
      at complete disposal of VM.

      Usually, System relevant classes as Error, TimeStamp, Stream and so on are
      considered WKI.

      Modules can declare their own WKIs so that they can safely retreive their own original
      live data that cannot be possibly transformed by scripts.

      Modules just need to declare a symbol adding the Symbol::setWKI clause, and the VM will create
      a proper entry on link step. WKI items are created for the module, but a safe copy is also
      saved in another location.

      WKI items have not to be exported, although they can.

      \note the data is not deep-copied. WKI are prevented to be chaged in their nature,
            but their content can still be changed. Please, notice that both flat items
            and classes are read-only from a script standpoint, while strings, arrays, objects
            and dictionaries can have their contents changed.

      \note The returned Well Known item is never to be de/referenced.
      \param name the WKI to be found
      \return 0 if a WKI with that name can't be found, a valid item pointer on success.
   */
   Item *findWKI( const String &name ) const;

   /** Returns a live global given the name of its symbol.
      Items exported by moduels becomes associated with an item that can be
      accessed (and then changed) by both scripts and modules extensions.

      This method returns a pointer to the item that stores the current value
      of the given global symbol, and it is exactly the item that the scripts
      using that symbol are seeing.

      The returned item is already dereferenced. Changing it will change all
      items referred by this symbol in all the modules.

      \note To create failsafe new instances of classes or to access critical
      functions exported by modules, use findWKI().

      \param name the symbol to be searched.
      \return the global item associated with the symbol if found, or 0 if the item does not exist.
   */
   Item *findGlobalItem( const String &name ) const;

   /** Returns the value of a local variable.
      This function searces for a variable in the local context, and eventually in the
      global context if not locally found.
      The function also parses accessors as the "." dot operator or the [] array access
      operator, but it doesn't support range accessors or function calls.

      As the value of object accessor may be synthezized on the fly, the method cannot
      return a physical pointer; the value of the variable is flat-copied in the data
      parameter.

      \param name the variable to be found.
      \param data where to store the value if found.
      \return true if the value can be found, false otherwise.
   */
   bool findLocalVariable( const String &name, Item &data ) const;

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

      Before calling this function, enough parameters must be pushed in the stack
      using pushParam() method.

      The paramCount parameter must be smaller or equal to the size of the stack,
      or an unblockable error will be raised.

      \param callable the item to be called.
      \param paramCount the number of elements in the stack to be considered parameters.
      \param mode the item call mode.
   */
   void callItem( const Item &callable, int32 paramCount )
   {
      callFrame( callable, paramCount );
      execFrame();
   }

   /** Shortcut for to call an item from a VM frame.
      Extension functions and VM/core functions meant to be called from the
      run() loop should use this function instead the callItem.

      The function prepares the VM to execute the desired item at the next run loop,
      as soon as the calling function returns.

      The caller should return immediately, or after a short cleanup, in case the
      call is succesful (and the function returns true).

      If the function needs to continue or do some post-processing after calling
      the callable item, it must install a return frame handler using returnHandler()
   */
   void callFrame( const Item &callable, int32 paramCount )
   {
      callable.readyFrame( this, paramCount );
   }

   /** Prepares a VM call frame with a return handler.
       When the code flow will return to the calling run(), or when it will enter
       run() for the first time, the code in callable will be executed.

       At termination, the vm will immediately call callbackFunc that may create new
       frames and return true, to ask the VM to continue the evaluation, or return
       false and terminate this frame, effectively "returning" from the callable.
   */
   void callFrame( const Item &callable, int32 paramCount, ext_func_frame_t callbackFunc )
   {
      callable.readyFrame( this, paramCount );
      m_currentContext->returnHandler( callbackFunc );
   }

   /** Prepare a frame for a function call */
   void prepareFrame( CoreFunc* cf, uint32 paramCount );

   /** Prepare a frame for an array call.

       The function oesn't check for the array to be callable because
       it's supposed that the check is on the caller.

       However, a debug assert is performed.
    */
   void prepareFrame( CoreArray* ca, uint32 paramCount );

   /** Executes an executable item in a coroutine.
     \param callable The callable routine.
     \param paramCount Number of parameters to be passed in the coroutine stack.
   */
   bool callCoroFrame( const Item &callable, int32 paramCount );

   /** Executes currently prepared frame. */
   void execFrame()
   {
      currentFrame()->m_break = true; // force to exit when meeting this limit
      m_currentContext->pc() = m_currentContext->pc_next();
      run();
   }

   StackFrame* currentFrame() const
   {
      return m_currentContext->currentFrame();
   }

   VMContext* currentContext() const
   {
      return m_currentContext;
   }

   /** Resets the return handler and prepares to call given external handler.
      This function prepares the VM to execute a return handler immediately
      after the calling function returns.

      The net effect is that, when called from inside an extension function,
      the given callback will be called by the VM as the very next operation,
      after checks on events, timings and context switches.

      The difference with callFrame is that the stack is unaffected, and
      the called function will have the same call frame as the caller.

      The difference with returnHandler is that the VM is instructed
      to execute the retun procedure (that will call the given call back
      function) immediately, while returnHandler just sets an handler
      for a future time when the readied call frame will be unrolled.

      This function can be safely called from inside the same callback
      function, so to create loops in which each call goes through
      VM checks for operation counts and events.

      The callbackFunc should behave as specified for returnHandler(),
      returning true if creating another frame with callFrame() or calling
      another return handler (or even itself) using callFrameNow().

      \note recallFrame() is a (more) efficient shortcut for
      using callFrameNow on the same calling function.

      \see returnHandler()

      \param callbackFunc the function to be called ASAP.
   */
   void callFrameNow( ext_func_frame_t callbackFunc );

   /** Prepare the VM to recall current return frame.
      Calling this method and returning true, an handler set with
      returnFrame() can instruct the VM to call itself again after
      having performed a loop check.

      \note This method must be called only inside extension functions,
            better if they are also return frame handlers.
      \see callFrameNow()
   */
   void recallFrame() { m_currentContext->pc_next() = m_currentContext->pc(); } // reset pc advancement

   /** Call an item in atomic mode.
      This method is meant to call the vm run loop from inside another vm
      run loop. When this is necessary, the inner call must end as soon as
      possible. The VM becomes unprehemptible; contexts are never switched,
      operation count limits (except for hard limits) are not accounted and
      any explicit try to ask for the VM to suspend, wait, yield or sleep
      raises an unblockable error.

      Things to be called in atomic mode are i.e. small VM operations
      overload methods, as the toString or the compare. All the rest should
      be performed using the callFrame mechanism.

      \throw CodeError if the item is not callable.
   */
   void callItemAtomic(const Item &callable, int32 paramCount );

   /** Installs a post-processing return frame handler.
      The function passed as a parmeter will receive a pointer to this VM.

      The function <b>MUST</b> return true if it performs another frame item call. This will
      tell the VM that the stack cannot be freed now, as a new call stack has been
      prepared for immediate execution. When done, the function will be called again.

      A frame handler willing to call another frame and not willing to be called anymore
      must first unininstall itself by calling this method with parameters set at 0,
      and then it <b>MUST return true</b>.

      A frame handler not installing a new call frame <b>MUST return false</b>. This will
      terminate the current stack frame and cause the VM to complete the return stack.
      \param callbackFunct the return frame handler, or 0 to disinstall a previously set handler.
   */
   void returnHandler( ext_func_frame_t callbackFunc ) {
      m_currentContext->returnHandler( callbackFunc );
   }

   ext_func_frame_t returnHandler() const
   {
      return m_currentContext->returnHandler();
   }


   /** Returns currently installed return handler, or zero if none.
      \return  currently installed return handler, or zero if none.
   */
   ext_func_frame_t returnHandler();

   /** Alias for void pushParameter() */
   void pushParam( const Item &item ) { m_currentContext->pushParam(item); }
   void pushParameter( const Item &item ) { m_currentContext->pushParam(item); }

   /** Adds some local space in the current context.
      \param amount how many local variables must be created
   */
   void addLocals( uint32 space )
   {
      m_currentContext->addLocals( space );
   }


   byte operandType( byte opNum ) const {
      return m_currentContext->code()[m_currentContext->pc() + 1 + opNum];
   }

   /** True if the VM is allowed to execute a context switch. */
   bool allowYield() { return m_allowYield; }

   /** Change turnover mode. */
   void allowYield( bool mode ) { m_allowYield = mode; }


   const ContextList *getCtxList() const { return &m_contexts; }
   const ContextList *getSleepingList() const { return &m_sleepingContexts; }

   /** Return from the last called subroutine.
      Usually used internally by the opcodes of the VM.
   */
   void callReturn()
   {
      // if the stack frame requires an end handler...
      // ... but only if not unrolling a stack because of error...
      if ( currentFrame()->m_endFrameFunc != 0 )
      {
         currentContext()->pc_next() = currentContext()->pc();
         // if the frame requires to stay here, return immediately
         if ( currentFrame()->m_endFrameFunc( this ) )
         {
            return;
         }
      }

      m_break = currentContext()->callReturn();

      // if we have nowhere to return...
      if( currentContext()->currentFrame() == 0 && ! m_break )
      {
         terminateCurrentContext();
      }
   }

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
      \param lmod the module id for which this class is being created.
      \param pt a ClassDef wise property table containing properties definition.
      \return a newly created property table ready to be added to a core class.
   */
   PropertyTable *createClassTemplate( LiveModule *lmod, const Map &pt );

   /** Publish a service.
      Will raise an error and return false if the service is already published.
      \param svr the service to be published on this VM.
      \throws CodeError on dupilcated names.
   */
   void publishService( Service *svr );


   /** Queries the VM for a published service.
      If a service with the given name has been published, the VM will return it;
      otherwise returns 0.
      \param name the service to be published on this VM.
      \return the required service or 0.
   */
   Service *getService( const String &name );


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
   virtual void reset();

   void limitLoops( uint32 l ) { m_opLimit = l; }
   uint32 limitLoops() const { return m_opLimit; }
   bool limitLoopsHit() const { return m_opLimit >= m_opCount; }

   void resetCounters();

   /** Performs a single VM step and return. */
   void step() {
      m_currentContext->pc() = m_currentContext->pc_next();
      m_bSingleStep = true;
      // stop next loop
      m_opNextCheck = m_opCount + 1;
      run();
   }

   void singleStep( bool ss ) { m_bSingleStep = ss; }
   bool singleStep() const { return m_bSingleStep; }

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
   void pushTry( uint32 landingPC ) { currentContext()->pushTry( landingPC ); }

   /** Pop a try position, eventually changing the frame to the handler. */
   void popTry( bool moveTo ) { currentContext()->popTry( moveTo ); }

   /** Elects a new context ready for execution.
      This method should be called by embedding applications that have performed
      a sleep operations right after the elapsed sleep time. The VM will elect
      the most suitable context for execution.

      On output, the VM will be ready to run (call the run() function); if no context
      is willing to run yet, there this method will set the eventSleep (hadSleepRequest()
      will return true), and the sleep request time will be propery set.
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

   /** Creates a reference to an item.
      The source item is turned into a reference which is passed in the
      target item. If the source is already a reference, the reference
      is just passed in the target.
      \param target the item that will accept reference to source
      \param source the source item to be referenced
   */
   void referenceItem( Item &target, Item &source );

   /** Set user data.
      VM is passed to every extension function, and is also quite used by the
      embedding application. To provide application specific per-vm data to
      the scripts, the best solution for embedding applications is usually
      to extend the VM into a subclass. Contrarily, middleware extensions,
      as, for example, script plugins for applications, may prefer to use the
      standard Falcon VM and use this pointer to store application specific data.

      The VM makes no assumption on the kind of user data. The data is not destroyed
      at VM destruction. If there is the need to destroy the data at VM destruction,
      then VM derivation seems a more sensible choice.

      \param ud the application specific user data.
   */
   void userData( void *ud ) { m_userData = ud; }

   /** Get the user data associated with this VM.
      \see userData( void * )
      \return the previously set user data or 0 if user data is not set.
   */
   void *userData() const { return m_userData; }

   /** Evaluate the item in a functional context.
      The function return either true or false, evaluating the item
      as a funcional token.
      # If the item is an array, it is recursively scanned.
      # If the item is a callable item (including callable arrayas),
        it is called and then the return value is evaluated in a
        non-functional context (plain evaluation).
      # In all the other cases, the item is evaluated in a non-functional
        context.

      More simply, if the item is callable is called, and is result
      is checked for Falcon truth value. If not, it's simply checked.

      The return value will be in regA(), and it's the "reduced" list
      of items.

      The function returns true if it has to stop for frame evaluation,
      false if the caller can loop. When the function returns true, the
      caller must immediately return to the calling frame; in case it
      needs to continue processing, it must install a frame return handler
      to be called back when the frame created by functionalEval is done.

      \param itm the item to be functionally evaluated.
      \param pcount the number of parameters in the VM to be treated as evaluation parameters.
      \param retArray true to force return of an evaluated array in case the input is not a Sigma sequence.
      \return false if the caller can proceed, true if it must return.
   */

   bool functionalEval( const Item &itm, uint32 pcount=0, bool retArray = true );

   /** Interrupts pending I/O on this machine from a separate thread.
      Interrupts compliant streams I/O and wait operations. The next I/O or
      the next wait, or the next compliant system blocking operation
      will cause an Interrupted exception to be raised.

      This method does not generate an "interrupt request" on the VM, that will
      keep on running until a blocking operation is encountered.

      This method can be safely called from other threads in the same application
      where the is currently executed, as well as from inside the same thread.
   */
   void interrupt() { m_systemData.interrupt(); }

   /** Returns true if the VM has been interrupted.
      If an asynchronous interrupt has been generated, this method will return true.

      This is meant to be called from those compliant operations that require polling,
      or just before starting waiting on interruptable code to avoid useless calculations,
      or right after to know if the VM has been interrupted during the computation.
      \param raise if true, prepare an Interrupted exception in the VM context if the VM has been
                   interrupted.
      \param reset if true, reset interrupted status now.
      \param dontCheck suppse interruption test is already done, and just raise the error and reset.
      \return true if the machine has been interrupted through interrupt() request.
   */
   bool interrupted( bool raise = false, bool reset = false, bool dontCheck = false );

   /** Get System Specific data.
      \returns system specific data bound with this machine.
   */
   virtual const Sys::SystemData &systemData() const { return m_systemData; }

   /** Get System Specific data (non const).
      \returns system specific data bound with this machine.
   */
   Sys::SystemData &systemData() { return m_systemData; }

   /** Exports a single symbol.
      This publishes a symbol on the global symbol map,
      and/or eventually to the WKS map.

      The function checks for the symbol to be exported and/or Well Known before
      actually performing the final export.
   */
   virtual bool exportSymbol( const Symbol *sym, LiveModule *mod );

   /** Exports all symbols in a module.
      To be called when changing the module publicity policy.
   */
   virtual bool exportAllSymbols( LiveModule *mod );

   /** Changes the status of launch-at-link mode */
   void launchAtLink( bool mode ) { m_launchAtLink = mode; }

   /** Returns the launch-at-link mode.
      This method returns true if the VM will launch the __main__ symbol of modules
      as soon as they are linked in the vm, or false otherwise.
   */
   bool launchAtLink() const { return m_launchAtLink; }

   /** Set current binding context.
      The current binding context is a dictionary containing
      a set of bound symbols and their value (referenced).

      Binding context is NOT GC masked, so it must exist
      elsewhere (i.e. in a live dictionary).

      The binding context is automatically removed at
      frame return.
   */
   void setBindingContext( CoreDict *ctx ) { m_currentContext->regBind() = ctx; }

   /** Return the value associated with a binding symbol.

      This function searches the given binding symbol name
      in the current binding context and in all the
      visible contexts (?).

      If the function returns 0, the symbol is unbound.
      \param bind The binding symbol name.
      \return A valid non-dereferenced binding value or 0 if the symbol is unbound.
   */
   Item *getBinding( const String &bind ) const;

   /** Return the value associated with a binding symbol, or creates one if not found.

      This function searches the given binding symbol name
      in the current binding context and in all the
      visible contexts (?). If the symbol is not found,
      it is created in the innermost visible context.

      If the function returns 0, then there is no visible context.
      \param bind The binding symbol name.
      \return A valid non-dereferenced binding value or 0 if there is no visible context.
   */
   Item *getSafeBinding( const String &bind );

   /** Set a binding value.

      This function sets a binding value in the current context.
      If a binding context has not been set, the function returns false.
      \param bind The binding symbol name.
      \param value The value to associate to this binding.
   */
   bool setBinding( const String &bind, const Item &value );


   typedef enum {
      lm_complete,
      lm_prelink,
      lm_postlink
   } t_linkMode;


   /** Request a constructor call after a call frame.
      If the preceding callFrame() was directed to an external function, requests the VM to treat
      the return value as an init() return, placing self() in regA() when all is done.
   */
   void requestConstruct() {
      if( m_currentContext->pc_next() == i_pc_call_external )
         m_currentContext->pc_next() = i_pc_call_external_ctor;
   }

   void setMetaClass( int itemID, CoreClass *metaClass )
   {
      fassert( itemID >= 0 && itemID < FLC_ITEM_COUNT );
      m_metaClasses[ itemID ] = metaClass;
   }

   CoreClass *getMetaClass( int itemID )
   {
      fassert( itemID >= 0 && itemID < FLC_ITEM_COUNT );
      return m_metaClasses[ itemID ];
   }

   /** Gets a slot for a given message.
      If the slot doesn't exist, 0 is returned, unless create is set to true.
      In that case, the slot is created anew and returned.
   */
   virtual CoreSlot* getSlot( const String& slotName, bool create = true );

   /** Removes and dereference a message slot in the VM.
   */
   virtual void removeSlot( const String& slotName );

   /** Used by the garbage collector to accunt for items stored as slot callbacks. */
   virtual void markSlots( uint32 mark );

   /** Comsume the currently broadcast signal.
      This blocks the processing of signals to further listener of the currently broadcasting slot.
      \return true if the signal is consumed, false if there was no signal to consume.
   */
   virtual bool consumeSignal();

   /** Declares an IDLE section.
      In code sections where the VM is idle, it is granted not to change its internal
      structure. This allow inspection from outer code, as i.e. the garbage collector.

      \note Calls VM baton release (just candy grammar).
      \see baton()
   */
   virtual void idle() { m_baton.release(); }

   /** Declares the end of an idle code section.
      \note Calls VM baton acquire (just candy grammar).
      \see baton()
   */
   virtual void unidle() { m_baton.acquire(); }

   /** Enable the Garbage Collector requests on this VM.

      If \b mode is false, the VM never checks for garbage collector requests to
      block operations for detailed inspections.

      If \b mode is true, the VM periodically tells the GC that it is ready for inspection,
      and it is forced to honor block requests when memory gets critically high.

      However, if the GC is disabled, the VM may be inspected if it volountarily enters
      the idle state (sleep or I/O calls).

      \param mode True to allow forced periodic inspections in case of need.
   */
   void gcEnable( bool mode );

   bool isGcEnabled() const;


   /** Setup the main script standard parameters and variables.

      This is an utility function filling the follwing global variables,
      provided they have been linked and are globally exported from some
      module in the VM:

      - script_name: the logical module name of the main module.
      - script_path: physical path of the main module.
      - args: filled with argc and argv.
   */
   virtual void setupScript( int argc, char** argv );


   /** Class automating idle-unidle fragments.
      This purely inlined class automathises the task of calling
      unidle() as a function putting the VM in idle (i.e. I/O function)
      returns.
   */
   class Pauser
   {
      VMachine *m_vm;
   public:
      inline Pauser( VMachine *vm ):
         m_vm( vm ) { m_vm->idle(); }

      inline ~Pauser() { m_vm->unidle(); }
   };

   /** Accessor to the VM baton.
      Used to serialize concurrent access to this VM.
   */
   const VMBaton& baton() const { return m_baton;  }

   /** Accessor to the VM baton.
      Used to serialize concurrent access to this VM.
   */
   VMBaton& baton() { return m_baton; }

   /** Send a message to the VMachine.

      If the virtual machine is currently idle, the message is immediately processed.

      Otherwise, it is posted to the main VM loop, and it is executed as soon as
      possible.

      The ownership of the message passes to the virtual machine, which will destroy
      it as the message is complete.

      The message is processed by broadcasting on the coresponding VM slot.
   */
   void postMessage( VMMessage *vm );

   /** Return current generation. */
   uint32 generation() const;

   /** Force a GC collection loop on the virtual machine.

      Waits for the GC loop to be completed. The virtual machine must
      be in non-idle mode when calling this function, as the idle ownership
      is directly transferred to the GC and then back to the calling VM
      without interruption.

      Normallym the GC will notify the VM back as soon as the mark loop is over;
      If the VM wants to wait for the free memory to be collected, set
      the parameter to true.
   */
   void performGC( bool bWaitForCollection = false );

   /** Increments the reference count for this VMachine. */
   void incref();

   /** Decrements the reference count for this virtual machine.
      If the count hits zero, the virtual machine is immediately destroyed.
   */
   void decref();

   /** Terminates this VM activity.

      The thread creating the VM or the VM owner shall call this to inform the system
         that the VM will not be actvive anymore.

      The VM will be destroyed immediately or as soon as all the other references
      are released.

      This fucntion also dereference the VM once (in the behalf of the VM original
      owner).
   */
   void finalize();

   /** Finalization callback function (used by MT) */
   void setFinalizeCallback( void (*finfunc)( VMachine* vm ) )
   {
      m_onFinalize = finfunc;
   }

   /** Executes the return frame as soon as the control reaches the VM main loop. 
      Can be used in conjunction with a fake call frame where the real work
      is done by the return frame function.

      \code
      static bool real_handler( VMachine *vm )
      {
         //do real work
         return false; // remove frame?
      }

      void extension_func( VMachine *vm )
      {
         vm->invokeReturnFrame( real_handler );  // prevent executing unexisting code in this frame.
         vm->addLocals( 5 );
         
         // configure local variables...

         // return, and let the VM call the real_handler
      }
      \endcode.

   */
   void invokeReturnFrame( ext_func_frame_t func ) { 
      createFrame( 0, func );
      m_currentContext->pc_next() = i_pc_call_external_return; 
   }

   /** Get the default application load path. */
   const String& appSearchPath() const { return m_appSearchPath; }

   /** Sets the default application load path (as seen by this vm). */
   void appSearchPath( const String &p ) { m_appSearchPath = p; }

   /** Call back on sleep requests.
      This method is called back when the virtual machine detects the
      need to perform a pause.

      The default VMachine version calls the system "sleep" routine,
      but the application may find something more interesting to
      do.

      @note The application should eventually call idle() and
      unidle() respectively at enter and exit of this callback if
      it doesn't use the VM while in this routine.

      @param seconds Number of seconds (and fractions) that the VM is
      idle.

      @throws InterruptedError if the wait was interrupted.
      @throws CodeError if the wait is < 0 (infinite) and there are
              no active contexts able to wake up this one.
    */

    virtual void onIdleTime( numeric seconds );
    //TODO: change signature next version
    bool replaceMe_onIdleTime( numeric seconds );

   /** Binds a late binding in the current context.

      As late bindings are known by name, only the name is necessary
      for the binding to complete.
      \param name the symbolic name of the binding.
      \param value the value to be associated with the late binding.
   */
   void bindItem( const String& name, const Item& value );

   /** Unbinds a late binding on the given target.
      If the lbind item is not a late binding, tgt is copied from this item.
      If lbind is a literal late binding, returns a non-literal late binding.
      If lbind is a non-literal late binding tries to resolve the late binding;
      in success placese the bound item (by value) in tgt.
   */

   void unbindItem( const String& name, Item &tgt ) const;

   /** Change the position of the next executed instruction. */
   void jump( uint32 pos )
   {
      m_currentContext->pc_next() = pos;
   }

   /** Expand a number of variables from the current value of an iterator.

       Uses the current VM pc_next position to decode \b count variables
    stored in NOP opcodes, storing the current value of the iterator
    into them.
    * */
   void expandTRAV( uint32 count, Iterator& iter );

   void breakRequest( bool mode ) { m_break = mode; }
   bool breakRequest() const { return m_break; }

   /** Returns the Random Number Generator */
   MTRand& getRNG(void) { return _mtrand; }

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
   friend void opcodeHandler_LDAS( register VMachine *vm );
   friend void opcodeHandler_SWCH( register VMachine *vm );
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

   friend void opcodeHandler_INCP( register VMachine *vm );
   friend void opcodeHandler_DECP( register VMachine *vm );

   friend void opcodeHandler_SHL( register VMachine *vm );
   friend void opcodeHandler_SHR( register VMachine *vm );
   friend void opcodeHandler_SHLS( register VMachine *vm );
   friend void opcodeHandler_SHRS( register VMachine *vm );
   friend void opcodeHandler_CLOS( register VMachine *vm );
   friend void opcodeHandler_PSHL( register VMachine *vm );
   friend void opcodeHandler_POWS( register VMachine *vm );
   friend void opcodeHandler_LSB( register VMachine *vm );
   friend void opcodeHandler_SELE( register VMachine *vm );
   friend void opcodeHandler_INDI( register VMachine *vm );
   friend void opcodeHandler_STEX( register VMachine *vm );
   friend void opcodeHandler_TRAC( register VMachine *vm );
   friend void opcodeHandler_WRT( register VMachine *vm );
   friend void opcodeHandler_STO( register VMachine *vm );
   friend void opcodeHandler_FORB( register VMachine *vm );
   friend void opcodeHandler_EVAL( register VMachine *vm );
   friend void opcodeHandler_OOB( register VMachine *vm );
   friend void opcodeHandler_TRDN( register VMachine *vm );
   friend void opcodeHandler_EXEQ( register VMachine *vm );
};


class FALCON_DYN_SYM VMachineWrapper
{
private:
   VMachine *m_vm;
public:
   VMachineWrapper():
      m_vm( new VMachine )
   {
   }

   VMachineWrapper( VMachine *host ):
      m_vm(host)
   {}

   ~VMachineWrapper()
   {
      m_vm->finalize();
   }

   VMachine *operator->() const
   {
      return m_vm;
   }

   VMachine *vm() const
   {
      return m_vm;
   }
};

}

#endif

/* end of vm.h */

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

#include <falcon/setup.h>
#include <falcon/baton.h>

#include <vector.h>

#define FALCON_VM_DFAULT_CHECK_LOOPS 5000

namespace Falcon {

/** The Falcon virtual machine.
*/
class FALCON_DYN_CLASS VMachine: public BaseAlloc
{

protected:

   Stream *m_stdIn;
   Stream *m_stdOut;
   Stream *m_stdErr;
   bool m_bhasStandardStreams;

   void internal_construct();
public:

   virtual ~VMachine();

public:

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

   /** Alias for void pushParameter() */
   void pushData( const Item &item ) { m_dataStacl->pushParam(item); }


   /** True if the VM is allowed to execute a context switch. */
   bool allowYield() { return m_allowYield; }

   /** Change turnover mode. */
   void allowYield( bool mode ) { m_allowYield = mode; }


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


   /** Changes the status of launch-at-link mode */
   void launchAtLink( bool mode ) { m_launchAtLink = mode; }

   /** Returns the launch-at-link mode.
      This method returns true if the VM will launch the __main__ symbol of modules
      as soon as they are linked in the vm, or false otherwise.
   */
   bool launchAtLink() const { return m_launchAtLink; }


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

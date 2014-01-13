/*
   FALCON - The Falcon Programming Language.
   FILE: vmprocess.cpp

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/vmprocess.cpp"

#include <falcon/cm/vmprocess.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/process.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>
#include <falcon/processor.h>
#include <falcon/modspace.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/function.h>
#include <falcon/stream.h>

namespace Falcon {
namespace Ext {


//====================================================
// Properties.
//


   /*#
      @property stdIn VMProcess
      @brief Creates an object mapped to the standard input of the Virtual Machine.

      The returned read-only stream is mapped to the standard input of the virtual
      machine hosting the script. Read operations will return the characters from the
      input stream as they are available. The readAvailable() method of the returned
      stream will indicate if read operations may block. Calling the read() method
      will block until some character can be read, or will fill the given buffer up
      the amount of currently available characters.

      The returned stream is a clone of the stream used by the Virtual Machine as
      standard input stream. This means that every transcoding applied by the VM is
      also available to the script, and that, when running in embedding applications,
      the stream will be handled by the embedder.

      As a clone of this stream is held in the VM, closing it will have actually no
      effect, except that of invalidating the instance returned by this function.

      Read operations will fail raising an I/O error.
   */


   /*#
      @property stdOut VMProcess
      @brief Creates an object mapped to the standard output of the Virtual Machine.

      The returned stream maps output operations on the standard output stream of
      the process hosting the script.

      The returned stream is a clone of the stream used by the Virtual Machine as
      standard output stream. This means that every transcoding applied by the VM is
      also available to the script, and that, when running in embedding applications,
      the stream will be handled by the embedder.

      As a clone of this stream is held in the VM, closing it will have actually no
      effect, except that of invalidating the instance returned by this function.

      Read operations will fail raising an I/O error.
   */


   /*#
      @property stdErr VMProcess
      @brief Creates an object mapped to the standard error of the Virtual Machine.

      The returned stream maps output operations on the standard error stream of
      the virtual machine hosting the script.

      The returned stream is a clone of the stream used by the Virtual Machine as
      standard error stream. This means that every transcoding applied by the VM is
      also available to the script, and that, when running in embedding applications,
      the stream will be handled by the embedder.

      As a clone of this stream is held in the VM, closing it will have actually no
      effect, except that of invalidating the instance returned by this function.

      Read operations will fail raising an I/O error.
   */
static void set_std_int(void* instance, const Item& value, int mode )
{
   static Class* strc = Engine::instance()->stdHandlers()->streamClass();

   Class* cls;
   void* data;
   if( ! value.asClassInst(cls, data) || ! cls->isDerivedFrom(strc ) )
   {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("Stream"));
   }

   Stream* stream = static_cast<Stream*>( cls->getParentData(strc,data) );
   if(stream != 0)
   {
      Process* prc = static_cast<Process*>(instance);

      switch(mode)
      {
      case 0: prc->stdIn(stream); break;
      case 1: prc->stdOut(stream); break;
      case 2: prc->stdErr(stream); break;
      }
   }
   else {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("Stream"));
   }
}


/*#
 @property stdOut VMProcess
 @brief overall access to VM
 */

static void get_stdIn(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   value.setUser( prc->stdIn()->handler(), prc->stdIn() );
}


/*#
 @property stdErr VMProcess
 @brief overall access to VM
 */
static void get_stdOut(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   value.setUser( prc->stdIn()->handler(), prc->stdOut() );
}

/*#
 @property stdIn VMProcess
 @brief overall access to VM
 */
static void get_stdErr(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   value.setUser( prc->stdIn()->handler(), prc->stdErr() );
}

static void set_stdIn(const Class*, const String&, void* instance, const Item& value )
{
  set_std_int(instance, value,0);
}

static void set_stdOut(const Class*, const String&, void* instance, const Item& value )
{
  set_std_int(instance, value,1);
}

static void set_stdErr(const Class*, const String&, void* instance, const Item& value )
{
  set_std_int(instance, value,2);
}

/*#
 @property id VMProcess
 @brief Returns the ID of this VM process.
 */
static void get_id(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   value = (int64) prc->id();
}

/*#
 @property current VMProcess
 @brief Returns an instance of the current VM process.
 */
static void get_current(const Class* cls, const String&, void*, Item& value )
{
   Process* prc = Processor::currentProcessor()->currentContext()->process();
   prc->incref();
   value = FALCON_GC_STORE(cls, prc);
}

/*#
 @property name VMProcess
 @brief Returns the name of the (main module of the) process

 This property assumes the value of the logical name of the module that
 leads the execution of the given process.

 The property can be invoked statically on the VMProcess class to
 obtain the name of the current process.
 */
static void get_name(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   if( prc == 0 )
   {
      // called statically
      prc = Processor::currentProcessor()->currentContext()->process();
   }

   Module* mod = prc->modSpace()->mainModule();
   if( mod != 0 )
   {
      value = FALCON_GC_HANDLE(new String(mod->name()));
   }
   else {
      value.setNil();
   }
}
/*#
 @property uri VMProcess
 @brief Returns the path of the (main module of the) process

 This property assumes the value of the logical name of the module that
 leads the execution of the given process.

 The property can be invoked statically on the VMProcess class to
 obtain the name of the current process.
 */
static void get_uri(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   if( prc == 0 )
   {
      // called statically
      prc = Processor::currentProcessor()->currentContext()->process();
   }

   Module* mod = prc->modSpace()->mainModule();
   if( mod != 0 )
   {
      value = FALCON_GC_HANDLE(new String(mod->uri()));
   }
   else {
      value.setNil();
   }
}

/*#
 @property error VMProcess
 @brief Termination error of this process.

 This returns the termination error of this process,
 or nil if there was none.

 The termination error can be accessed during cleanup
 routines; it will be nil (usually) during normal
 execution.
 */
static void get_error(const Class*, const String&, void* instance, Item& value )
{
   Process* prc = static_cast<Process*>(instance);
   if( prc == 0 )
   {
      // called statically
      prc = Processor::currentProcessor()->currentContext()->process();
   }

   if ( prc->error() != 0 )
   {
      value.setUser( prc->error()->handler(), prc->error() );
   }
   else {
      value.setNil();
   }
}

/*
 @method quit VMProcess
 @brief Terminates all the processes currently active in the host Virtual Machine.
 */
/*static void vm_quit( VMContext*, int32 )
{
   VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
   vm->quit();
}
*/

namespace CVMProcess {

/*#
  @method getIs Process
  @brief Returns an array containing all the translable strings in the system.
 */
FALCON_DECLARE_FUNCTION( getIs, "" )
void Function_getIs::invoke( VMContext* ctx, int32 )
{
   Process* proc = static_cast<Process*>(ctx->self().asInst());
   ModSpace* ms = proc->modSpace();
   class Rator: public ModSpace::IStringEnumerator
   {
   public:
      Rator(ItemArray* arr): m_arr(arr) {}
      virtual ~Rator() {}
      virtual bool operator()(const String& string )
      {
         m_arr->append( FALCON_GC_HANDLE( new String(string) ) );
         return true;
      }
   public:
      ItemArray* m_arr;
   };

   ItemArray* arr = new ItemArray;
   Rator rator(arr);
   ms->enumerateIStrings(rator);

   ctx->returnFrame(FALCON_GC_HANDLE(arr));
}

/*#
  @method setTT Process
  @brief Sets the translation table for this process.
  @param table A string => string dictionary of translations.
  @optparam add if true, the translations in the table will be added to the existing one.

  After the table is loaded, the i"" strings will be translated accordingly with
  the given table throughout all the process.
 */
FALCON_DECLARE_FUNCTION( setTT, "table:D, add:[B]" )
void Function_setTT::invoke( VMContext* ctx, int32 )
{
   Item* i_table = ctx->param(0);
   Item* i_add = ctx->param(1);
   if( i_table == 0 || ! i_table->isDict() )
   {
      throw paramError(__LINE__, SRC);
   }

   bool add = i_add == 0 ? false : i_add->isTrue();
   Process* proc = static_cast<Process*>(ctx->self().asInst());
   bool result = proc->setTranslationsTable( i_table->asDict(), add );
   if( ! result )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra( "not S=>S") );
   }

   ctx->returnFrame();
}


/*#
  @method pushCleanup Process
  @brief Adds an entity that will be evaluated at cleanup.
  @param code The code to be evaluated

  This methods adds an entity (normally a callable item, as a function,
  method, callable array or syntactic tree) for delayed execution at
  process termination.

  This can be used to add routines that clear the state of the host system
  no matter what happens during the process execution.

  If the process is terminated due to an error thrown and uncaught in the
  main context, the error will be available to the outer OS or embedding
  program after all the cleanup sequences are completed.

  In case a cleanup sequence raises an error, the one that was raised
  in the main context will be discarded.

  Cleanup sequences can push new cleanup sequences during their execution;
  the new cleanup sequences will be performed as all the previously pushed
  sequences are completed, that is, as the process is about to end.

  @note Added cleanup code is invoked in reverse order (last-to-first).
 */
FALCON_DECLARE_FUNCTION( pushCleanup, "code:C" )
void Function_pushCleanup::invoke( VMContext* ctx, int32 )
{
   Item* i_code = ctx->param(0);
   if( i_code == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   Process* proc = static_cast<Process*>(ctx->self().asInst());
   proc->pushCleanup(*i_code);
   ctx->returnFrame();
}



/*#
  @method clearError Process
  @brief Removes the termination error.

  If the process terminated with an error, this method
  will clear it.
 */
FALCON_DECLARE_FUNCTION( clearError, "" )
void Function_clearError::invoke( VMContext* ctx, int32 )
{
   Process* proc = static_cast<Process*>(ctx->self().asInst());
   proc->clearError();
   ctx->returnFrame();
}



static void internal_pdata( PData* pd, VMContext* ctx, Function* caller )
{
   // Get the ID of the persistent data
   Falcon::Item *i_id = ctx->param( 0 );
   Falcon::Item *i_func = ctx->param( 1 );

   // parameter sanity check.
   if ( i_id == 0 || ! i_id->isString() ||
         ( i_func != 0 && ! (i_func->isCallable() || i_func->isNil() || i_func->isTreeStep()) )
      )
   {
      throw caller->paramError();
   }


   Item target;
   const String& id = *i_id->asString();
   bool success = pd->get( id, target, true );

   // Not found?
   if( ! success )
   {
      // should we initialize the data?
      if ( i_func != 0 )
      {
         if( i_func->isNil() )
         {
            pd->remove(id);
            ctx->returnFrame(target);
         }
         else {
            ClassVMProcess* clw = static_cast<ClassVMProcess*>(caller->methodOf());
            ctx->pushCode( clw->m_stepAfterPersist );
            ctx->addLocals(1);
            ctx->local(0)->setOpaque("PData", pd);
            ctx->callItem(*i_func);
         }
      }
      else
      {
         throw FALCON_SIGN_XERROR( AccessError, e_object_not_found, .extra(id) );
      }
   }
   // otherwise, all ok
   else {
      ctx->returnFrame(target);
   }
}


static void internal_pdata_set( PData* pd, VMContext* ctx, Function* caller )
{
   // Get name of the file.
   Falcon::Item *i_id = ctx->param( 0 );
   Falcon::Item *i_item = ctx->param( 1 );

   // parameter sanity check.
   if ( i_id == 0 || ! i_id->isString() || i_item == 0 )
   {
      throw caller->paramError();
   }

   const String& id = *i_id->asString();

   if( i_item->isNil() )
   {
      pd->remove( id );
   }
   else {
      pd->set( id, *i_item );
   }

   ctx->returnFrame();
}

/*#
  @method pdata Process
  @brief Adds or remove processor-wide persistent data.
  \param id The symbolic ID name of the data to be set.
  \optparam setup a callable that will be invoked to generate the item if not found.
  \return the persistent item.
  \raise AccessError if the given id is not associated with any object.

   This method restores vm-context process-context specific
   persistent data that was previously saved, possibly by another VM process
   (i.e. another script).

   An optional @b code parameter is evaluated in case the data under
   the given @b id is not found; the evaluation result is
   then d in the persistent data slot, as if @a process.pdataSet was
   called with the same @b id to save the data, and is then returned to
   the caller of this method.

   If setup is nil, the persistent object will be removed, if found.

   If the object is not found and @b setup is not given, an @a AccessError
   will be raised.
 */
FALCON_DECLARE_FUNCTION( pdata, "id:S,setup:[C]" )
FALCON_DEFINE_FUNCTION_P1(pdata)
{
   PData* pd = Processor::currentProcessor()->pdata();
   internal_pdata( pd, ctx, this );
}


/*#
  @method pdataSet Process
  @brief Unconditionally sets processor-wide data
  \param id The symbolic ID name of the data to be set.
  \param value the value to be set.


   This method saves Virtual Machine process-specific data, to be
   retrieved, eventually by another VM process at a later
   execution under the same O/S context (usually, a thread or a process).

   Persistent data is identified by an unique ID. An application
   can present different persistent data naming them differently.

   @note The id can be any a valid string, including an empty string.
   So, "" can be used as a valid persistent key.

   Setting the item to nil effectively removes the entry from the
   persistent storage. However, the data will be reclaimed (and finalized)
   during the next garbage collection loop.
 */
FALCON_DECLARE_FUNCTION( pdataSet, "id:S,value:X" )
FALCON_DEFINE_FUNCTION_P1(pdataSet)
{
   PData* pd = Processor::currentProcessor()->pdata();
   internal_pdata_set( pd, ctx, this );
}


/*#
  @method vmdata Process
  @brief Adds or remove virtual-machine wide persistent data.
  \param id The symbolic ID name of the data to be set.
  \optparam setup a callable that will be invoked to generate the item if not found.
  \return the persistent item.
 */
FALCON_DECLARE_FUNCTION( vmdata, "id:S,setup:[C]" )
FALCON_DEFINE_FUNCTION_P1(vmdata)
{
   Process* proc = static_cast<Process*>(ctx->self().asInst());
   PData* pd = proc->vm()->pdata();
   internal_pdata( pd, ctx, this );
}


/*#
  @method vmdataSet Process
  @brief Unconditionally sets virtual-machine wide persistent data.
  \param id The symbolic ID name of the data to be set.
  \param value the value to be set.
 */
FALCON_DECLARE_FUNCTION( vmdataSet, "id:S,value:X" )
FALCON_DEFINE_FUNCTION_P1(vmdataSet)
{
   Process* proc = static_cast<Process*>(ctx->self().asInst());
   PData* pd = proc->vm()->pdata();
   internal_pdata_set( pd, ctx, this );
}


/*#
  @method edata Process
  @brief Adds or remove falcon engine (O/S process) wide persistent data.
  \param id The symbolic ID name of the data to be set.
  \optparam setup a callable that will be invoked to generate the item if not found.
  \return the persistent item.
 */
FALCON_DECLARE_FUNCTION( edata, "id:S,setup:[C]" )
FALCON_DEFINE_FUNCTION_P1(edata)
{
   static PData* pd = Engine::instance()->pdata();
   internal_pdata( pd, ctx, this );
}


/*#
  @method edataSet Process
  @brief Unconditionally sets falcon engine (O/S process) wide persistent data.
  \param id The symbolic ID name of the data to be set.
  \param value the value to be set.
 */
FALCON_DECLARE_FUNCTION( edataSet, "id:S,value:X" )
FALCON_DEFINE_FUNCTION_P1(edataSet)
{
   static PData* pd = Engine::instance()->pdata();
   internal_pdata_set( pd, ctx, this );
}

class PStepAfterPersist: public PStep
{
public:
   inline PStepAfterPersist() { apply = apply_; }
   inline virtual ~PStepAfterPersist() {}
   virtual void describeTo( String& target ) const { target = "VMProcess::StepAfterPersist"; }

   static void apply_(const PStep*, VMContext* ctx)
   {
      // we have the object to be saved in the persist data in top,
      // and the frame is that of the persist() method
      Falcon::Item *i_id = ctx->param( 0 );
      const String& src = *i_id->asString();
      fassert( String("PData") == ctx->local(0)->asOpaqueName());
      PData* pd = static_cast<PData*>(ctx->local(0)->asOpaque());
      Item top = ctx->topData();
      pd->set( src, top );
      ctx->returnFrame(top);
   }
};

}

//==========================================================================
// Main class
//==========================================================================

ClassVMProcess::ClassVMProcess():
         Class("VMProcess")
{
   m_bHasSharedInstances = true;
   m_stepAfterPersist = new CVMProcess::PStepAfterPersist;


   addProperty( "current", &get_current, 0, true ); //static

   addProperty( "stdIn", &get_stdIn, &set_stdIn );
   addProperty( "stdOut", &get_stdOut, &set_stdOut );
   addProperty( "stdErr", &get_stdErr, &set_stdErr );
   addProperty( "id", &get_id );
   addProperty( "error", &get_error );

   addProperty( "name", &get_name, 0, true );
   addProperty( "uri", &get_uri, 0, true );

   addMethod( new CVMProcess::Function_getIs );
   addMethod( new CVMProcess::Function_setTT );
   addMethod( new CVMProcess::Function_pushCleanup );
   addMethod( new CVMProcess::Function_clearError );

   addMethod( new CVMProcess::Function_pdata );
   addMethod( new CVMProcess::Function_pdataSet );
   addMethod( new CVMProcess::Function_vmdata );
   addMethod( new CVMProcess::Function_vmdataSet );
   addMethod( new CVMProcess::Function_edata );
   addMethod( new CVMProcess::Function_edataSet );

}

ClassVMProcess::~ClassVMProcess()
{
   delete m_stepAfterPersist;
}

void* ClassVMProcess::createInstance() const
{
   VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
   return new Process(vm);
}

void ClassVMProcess::dispose( void* instance ) const
{
   Process* prc = static_cast<Process*>(instance);
   prc->decref();
}

void* ClassVMProcess::clone( void* ) const
{
   return 0;
}

void ClassVMProcess::gcMarkInstance( void* instance, uint32 mark ) const
{
   Process* prc = static_cast<Process*>(instance);
   prc->gcMark(mark);
}

bool ClassVMProcess::gcCheckInstance( void* instance, uint32 mark ) const
{
   Process* prc = static_cast<Process*>(instance);
   return prc->currentMark() >= mark;
}

}
}

/* end of gc.cpp */

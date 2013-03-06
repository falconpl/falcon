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
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/accesstypeerror.h>
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
 @brief Returns the ID of this VM process.
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

/*
 @method quit VM
 @brief Terminates all the processes currently active in the host Virtual Machine.
 */
/*static void vm_quit( VMContext*, int32 )
{
   VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
   vm->quit();
}
*/

ClassVMProcess::ClassVMProcess():
         Class("VMProcess")
{
   addProperty( "current", &get_current, 0, true ); //static

   addProperty( "stdIn", &get_stdIn, &set_stdIn );
   addProperty( "stdOut", &get_stdOut, &set_stdOut );
   addProperty( "stdErr", &get_stdErr, &set_stdErr );
   addProperty( "id", &get_id );
   addProperty( "name", &get_name, 0, true );
   addProperty( "uri", &get_uri, 0, true );

}

ClassVMProcess::~ClassVMProcess()
{

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

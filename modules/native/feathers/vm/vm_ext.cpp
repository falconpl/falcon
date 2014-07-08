/* FALCON - The Falcon Programming Language.
 * FILE: vm_ext.cpp
 * 
 * Interface to the virtual machine
 * Main module file, providing the module object to the Falcon engine.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Wed, 06 Mar 2013 17:24:56 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: The above AUTHOR
 * 
 * See LICENSE file for licensing details.
 */

#undef SRC
#define SRC "modules/native/feathers/vm/vm_ext.cpp"

/*#
   @beginmodule feathers.vm
*/

#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>

#include <falcon/error.h>
#include <falcon/stderrors.h>

#include <falcon/vm.h>
#include <falcon/processor.h>
#include <falcon/stream.h>

#include "vm_ext.h"

namespace Falcon { 
    namespace Ext {

       static void set_std_int(const Item& value, int mode )
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
             VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
             switch(mode)
             {
             case 0: vm->stdIn(stream); break;
             case 1: vm->stdOut(stream); break;
             case 2: vm->stdErr(stream); break;
             }
          }
          else {
             throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("Stream"));
          }
       }


       /*#
        @property stdOut VM
        @brief overall access to VM
        */

       static void get_stdIn(const Class*, const String&, void*, Item& value )
       {
          VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
          value.setUser( vm->stdIn()->handler(), vm->stdIn() );
       }


       /*#
        @property stdErr VM
        @brief overall access to VM
        */
       static void get_stdOut(const Class*, const String&, void*, Item& value )
       {
          VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
          value.setUser( vm->stdIn()->handler(), vm->stdOut() );
       }

       /*#
        @property stdIn VM
        @brief overall access to VM
        */
       static void get_stdErr(const Class*, const String&, void*, Item& value )
       {
          VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
          value.setUser( vm->stdIn()->handler(), vm->stdErr() );
       }

       static void set_stdIn(const Class*, const String&, void*, const Item& value )
       {
         set_std_int(value,0);
       }

       static void set_stdOut(const Class*, const String&, void*, const Item& value )
       {
         set_std_int(value,1);
       }

       static void set_stdErr(const Class*, const String&, void*, const Item& value )
       {
         set_std_int(value,2);
       }

       /*#
        @property processes VM
        @brief Count of currently active processes.
        */
       static void get_processes(const Class*, const String&, void*, Item& value )
       {
          VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
          value = (int64) vm->getProcessorCount();
       }


       /*#
        @method quit VM
        @brief Terminates all the processes currently active in the host Virtual Machine.
        */
       static void vm_quit( VMContext*, int32 )
       {
          VMachine* vm = Processor::currentProcessor()->currentContext()->vm();
          vm->quit();
       }

       /*#
        @object VM
        @brief Access to the virtual machine running the process of this script.
        */
       ClassVM::ClassVM():
                Class("%VM")
       {
          // we don't need an object
          m_bIsFlatInstance = true;
          addProperty( "stdIn", &get_stdIn, &set_stdIn );
          addProperty( "stdOut", &get_stdOut, &set_stdOut );
          addProperty( "stdErr", &get_stdErr, &set_stdErr );
          addProperty( "processes", &get_processes );

          addMethod( "quit", &vm_quit, "" );
       }

       ClassVM::~ClassVM()
       {

       }

       void* ClassVM::createInstance() const
       {
          return 0;
       }

       void ClassVM::dispose( void* ) const
       {
          // nothing to do
       }

       void* ClassVM::clone( void* ) const
       {
          // nothing to do
          return 0;
       }

       bool ClassVM::op_init( VMContext* ctx, void* , int32 ) const
       {
          // we have nothing to configure
          ctx->topData() = Item(this,0);
          return false;
       }
    }
} // namespace Falcon::Ext

/* end of vm_ext.cpp */


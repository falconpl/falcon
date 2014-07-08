/* FALCON - The Falcon Programming Language.
 * FILE: vm_ext.h
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

#ifndef FALCON_EXT_VM_H
#define FALCON_EXT_VM_H

#include <falcon/module.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/class.h>

namespace Falcon { 
    namespace Ext {

       class ClassVM: public ::Falcon::Class
       {
       public:
          ClassVM();
          virtual ~ClassVM();

          void* createInstance() const;
          void dispose( void* instance ) const;
          void* clone( void* instance ) const;
          bool op_init( VMContext*, void* , int32 ) const;
       };
    }
} // namespace Falcon::Ext

#endif

/* end of vm_ext.h */


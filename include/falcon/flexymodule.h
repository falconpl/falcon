/*
   FALCON - The Falcon Programming Language.
   FILE: flextmodule.h

   Falcon flexible module prototype.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 24 Jun 2008 18:24:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_FLEXY_MODULE_H
#define FLC_FLEXY_MODULE_H

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/module.h>

namespace Falcon {

/** Flexible module class.
   Normally, modules are read-only from a VM standpoint.

   However, modules willing to be dynamically changed by the VM can declare their
   disponibility by inheriting from this class.

   Such modules should be given only to the VM they are linked first. Modules
   created by a VM for internal needs (i.e. by the reflexive system) are
   of this type.

   The most interesting feature of this kind of modules is that of being
   able to provide new symbols at runtime. See the onSymbolRequest() method.
*/
class FALCON_DYN_CLASS FlexyModule: public Module
{
public:
   /** Mark this module as a Flexy module. */
   virtual bool isFlexy() const { return true; }

   /** Describe this module constness.
      Flexy modules should not normally be cached by applications willing
      to cache modules on a global map and serving to the VMs through a
      modified module loader.

      However, if the module guarantees that it doesn't modify the interface
      it presents when loaded by other vms, i.e. by not creating new symbols
      or changing the valence and internals of existing ones, they can
      change this mehtod to return true.

      Application are advised that flexy modules returning true in isConst
      are safe to be shared among different VMs and can be cached at
      application level.
   */
   virtual bool isConst() const { return false; }

   /** Provides symbol on VM dynamic request.

      The onSymbolRequest() method is a callback that the VM lanunches on all the
      regiestered modules when a module wants to link a symbol that is not found
      in the global export table. The flexy module then able to provide the symbol
      to the VM, and to get it linked just-in-time, on script request.

      This makes room for full by-request import, as supported by many scripting
      languages.

      The flexy module may respect this call and still be const; the module may
      create the needed symbol locally and refuse to export them. Then, on VM
      request, it can provide the local symbol returing them through this
      callback. Doing so, it is granted that the module won't be changed in runtime,
      and this makes possible to share a flexy module across several VMs too.
   */

   virtual Symbol *onSymbolRequest( const String &name ) = 0;
};

}

#endif

/* end of flexymodule.h */

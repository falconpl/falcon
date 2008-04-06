/*
   FALCON - The Falcon Programming Language.
   FILE: flc_runtime.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer ago 18 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_RUNTIME_H
#define flc_RUNTIME_H

#include <falcon/setup.h>
#include <falcon/modloader.h>
#include <falcon/common.h>
#include <falcon/genericvector.h>
#include <falcon/genericmap.h>
#include <falcon/basealloc.h>

namespace Falcon {

class VMachine;

/** Map of module names-> modules.
    ( const String *, Module * )
*/
class FALCON_DYN_CLASS ModuleMap: public Map
{
public:
   ModuleMap();
};

class FALCON_DYN_CLASS ModuleVector: public GenericVector
{
public:
   ModuleVector();
   Module *moduleAt( uint32 pos ) const { return *(Module**) at(pos); }
};

/** Runtime code representation.

A runtime object represents a set of module that constitutes the existing runtime
library. Two basic operations can be done on a runtime set: module addition and
linking.

A runtime module is given a runtime error manager to inform the calling application
about errors during module addition and linking (mainly, load failure ).

If provided with a ModuleLoader instance, the Runtime instance will ask the module
loader to load any module that the given modules require to import, unless already
present in the module set; else, the
calling application must provide the modules to the Runtime one by one. This allows
to put the embedding application in complete control of what modules may be loaded
and how, and allows it to provide its own internal modules instead of the required
ones.

At the moment, the modules must be pre-compiled in hsc format, but in future
the module loader may detect if a module is given in source and compile it if the
hsc is not present; this is a behavior bound in the module loader.

\todo In future, this class may support serialization to store the linked modules on
disk, so that it may become a somewhat complete linker.

*/


class FALCON_DYN_CLASS Runtime: public BaseAlloc
{
   ModuleMap m_modules;
   ModuleVector m_modvect;
   ModuleLoader *m_loader;
   VMachine *m_provider;

public:

   /** Creates the runtime and allows module load requests.

      Using this version of the constructor, the runtime will try to resolve module
      load requests that are embedded in each module before trying to link given
      modules.

      An error manager can be specified later on with errorHandler().

      The error manager is used to feedback the calling application or the user with
      rilevant error codes (i.e. the module loading faulire, symbol duplication or
      missing externals).

      Neither the error manager nor the module loader are deleted at Runtime object
      deletion.

      \param loader a module loader that will be used to load modules when they are
         required by other modules at link() time.
      \param provider a virtual machine where the runtime must be linked later on.

   */
   Runtime( ModuleLoader *loader, VMachine *provider = 0 );


   /** Creates the runtime.

      Using this version of the constructor, the runtime will not load any module that
      may be defined in the dependency list during the link phase; the calling
      application must load or provide the modules on its behalf and feed them in
      the link() method before the subsequesing modules require some symbols as
      external.

      An error manager can be specified later on with errorHandler().

      The error manager is used to feedback the calling application or the user with
      rilevant error codes (i.e. the module loading faulire, symbol duplication or
      missing externals).

      The error manager is not deleted at Runtime object deletion.
   */
   Runtime();

   /** Destroy this Runtime.

      When a runtime is destroyed, modules linked by this runtime are de-referenced
      and eventually destroyed. To save modules after their last runtime released them,
      add an Module::incref() call soon after their creation.
   */
   ~Runtime();

   /** Tell wether this runtime tries to interpret the module load requests or just denies them.
      \return true if the runtime fulfills module requests with a provided ModuleLoader,
              false otherwise.
   */

   bool willLoad() { return m_loader != 0 ; }

   /** Adds a module to the Runtime library.

      This function adds a module to the runtime library and then it
      links it in two steps.

      First, link() method is called to fill the module global variables table,
      (which may be empty), by importing external symbols from the already
      added modules. Then all exported symbol is uploaded in the global symbols
      table (m_gsyms).

      Module symbol exporting overrides previous export requests. In example, if
      the rtl module (providing and exporting \b printl function ) and then another
      module exporting a symbol printl is added, all the subsequent references to
      printl will refer to the latter definition. However, modules importing printl()
      before the latter override will still refer to the original printl(). To override
      \b all instance of an already defined function, just assign it to something else
      everywhere in the executed module(s).

      When a module is added to the runtime, its reference count is incremented, and
      a record containing its ID (relative position) is added. The ID will soft-link
      the module with its own global variable representation in the executing VM (i.e. Module N
      will have global variable vector number N, module K will refer to variables in the K vector
      and so on).

   */
   bool addModule( Module *mod );

   /** Return the symbol tably containing the global symbols.
      As a symbol table is just a symbol map that self-deletes owned symbol,
      and as this map just holds a reference to some symbols inside the modules
      contained in the runtime, the map must not destroy the symbols at runtime
      destruction. Symbols are owned by symbols table, which are owned by
      their modules. However, as modules cannot be deleted while they are
      still owned by some Runtime, this references are guaranteed to stay valid
      at least untill this Runtime is destroyed.
   */

   /** Return the modules linked in this runtime. */
   const ModuleMap *moduleMap() const { return &m_modules; }

   /** Return the modules linked in this runtime. */
   const ModuleVector *moduleVector() const { return &m_modvect; }

   /** Returns a module with the given name.
      Or zero if the module with the required name is not found.
   */
   Module *findModule( const String &name )
   {
      Module **modp = (Module **) m_modules.find( &name );
      if ( modp != 0 )
         return *modp;
      return 0;
   }

   /** Return the nth module. */
   Module *findModuleByID( uint32 id ) {
      return m_modvect.moduleAt( id );
   }

   /** Return the amount of modules in this runtime. */
   uint32 size() { return m_modvect.size(); }


   /** Loads the given module and adds it to the runtime.
      This is actually a just shortcut to load a module
      with a given logical name. The name is resolved by the module loader,
      and if load is succesfull, then the module is added to the runtime.
      \param name the logical name of the module to be loaded
      \param parent the logical name of the parent module, if any
      \return true on success, false on load or dependency resolution errors.
   */
   bool loadName( const String &name, const String &parent = "" );

   /** Loads the given module and adds it to the runtime.
      This is actually a just shortcut to load a module
      from a given file. The name is resolved by the module loader,
      and if load is succesfull, then the module is added to the runtime.

      \param file the file name of the module to be loaded
      \return true on success, false on load or dependency resolution errors.
   */
   bool loadFile( const String &file );
};


}

#endif

/* end of flc_runtime.h */

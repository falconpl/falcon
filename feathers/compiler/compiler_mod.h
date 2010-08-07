/*
   FALCON - The Falcon Programming Language.
   FILE: compiler_mod.h

   Compiler interface modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler interface modules
*/

#ifndef flc_compiler_mod_H
#define flc_compiler_mod_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/modloader.h>
#include <falcon/string.h>
#include <falcon/vmmaps.h>
#include <falcon/falcondata.h>
#include <falcon/coreobject.h>
#include <falcon/intcomp.h>

namespace Falcon {

namespace Ext {

CoreObject* CompilerIfaceFactory( const CoreClass *cls, void *, bool );

class CompilerIface: public CoreObject
{
protected:

   ModuleLoader m_loader;
   String m_sourceEncoding;
   bool m_bLaunchAtLink;

public:
   CompilerIface( const CoreClass* cls );
   CompilerIface( const CoreClass* cls, const String &path );

   virtual ~CompilerIface();

   const ModuleLoader &loader() const { return m_loader; }
   ModuleLoader &loader() { return m_loader; }

   bool launchAtLink() const { return m_bLaunchAtLink; }
   void launchAtLink( bool l ) { m_bLaunchAtLink = l; }

   // nothing to mark
   virtual void gcMark( uint32 mk ) {}

   // Override set property
   virtual bool setProperty( const String &prop, const Item &value );

   // Override get property
   virtual bool getProperty( const String &key, Item &ret ) const;

   // we don't provide a clone() method
   virtual CoreObject *clone() const  { return 0; };
};


CoreObject* ICompilerIfaceFactory( const CoreClass *cls, void *, bool );

/** Interactive version of the compiler */
class ICompilerIface: public CompilerIface
{
protected:
   InteractiveCompiler* m_intcomp;

   /** The VM private for this compiler */
   VMachine *m_vm;

public:
   ICompilerIface( const CoreClass* cls );
   ICompilerIface( const CoreClass* cls, const String &path );
   virtual ~ICompilerIface();

   // Override set property
   virtual bool setProperty( const String &prop, const Item &value );
   // Override get property
   virtual bool getProperty( const String &key, Item &ret ) const;
   
   VMachine* vm() const { return m_vm; }
   InteractiveCompiler* intcomp() const { return m_intcomp; }
};


class ModuleCarrier: public FalconData
{
   LiveModule *m_lmodule;

public:
   ModuleCarrier( LiveModule *m_module );
   virtual ~ModuleCarrier();

   const Module *module() const;
   LiveModule *liveModule() const { return m_lmodule; }

   virtual FalconData *clone() const;
   virtual void gcMark( uint32 mk );

   // we don't provide a clone() method
};


}

}

#endif

/* end of compiler_mod.h */

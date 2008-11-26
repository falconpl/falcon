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
#include <falcon/userdata.h>
#include <falcon/flcloader.h>
#include <falcon/string.h>
#include <falcon/vmmaps.h>

namespace Falcon {

namespace Ext {

class CompilerIface: public UserData
{
   FlcLoader m_loader;
   String m_sourceEncoding;

public:
   CompilerIface();
   CompilerIface( const String &path );

   virtual ~CompilerIface();

   const FlcLoader &loader() const { return m_loader; }
   FlcLoader &loader() { return m_loader; }

   virtual void getProperty( CoreObject *vm, const String &propName, Item &prop );
   virtual void setProperty( CoreObject *vm, const String &propName, const Item &prop );

   // we don't provide a clone() method
   virtual FalconData *clone() const { return 0; }
};

class ModuleCarrier: public FalconData
{
   LiveModule *m_lmodule;

public:
   ModuleCarrier( LiveModule *m_module );
   virtual ~ModuleCarrier();

   const Module *module() const { return m_lmodule->module(); }
   LiveModule *liveModule() const { return m_lmodule; }

   virtual FalconData *clone() const;
   virtual void gcMark( VMachine *vm );

   // we don't provide a clone() method
};


}

}

#endif

/* end of compiler_mod.h */

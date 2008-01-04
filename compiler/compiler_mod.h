/*
   FALCON - The Falcon Programming Language.
   FILE: compiler_mod.h
   $Id: compiler_mod.h,v 1.2 2007/08/11 19:02:32 jonnymind Exp $

   Compiler interface modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
   CoreObject *m_owner;

public:
   CompilerIface( CoreObject *mod );
   CompilerIface( CoreObject *mod, const String &path );

   virtual ~CompilerIface();

   const FlcLoader &loader() const { return m_loader; }
   FlcLoader &loader() { return m_loader; }

   virtual bool isReflective();
   virtual void getProperty( const String &propName, Item &prop );
   virtual void setProperty( const String &propName, Item &prop );

   // we don't provide a clone() method
};

class ModuleCarrier: public UserData
{
   LiveModule *m_lmodule;
   
public:
   ModuleCarrier( LiveModule *m_module );
   virtual ~ModuleCarrier();

   const Module *module() const { return m_lmodule->module(); }
   LiveModule *liveModule() const { return m_lmodule; }

   virtual UserData *clone();
   virtual void gcMark( MemPool *mp );

   // we don't provide a clone() method
};


}

}

#endif

/* end of compiler_mod.h */

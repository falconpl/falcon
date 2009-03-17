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

namespace Falcon {

namespace Ext {

CoreObject* CompilerIfaceFactory( const CoreClass *cls, void *, bool );

class CompilerIface: public CoreObject
{
   ModuleLoader m_loader;
   String m_sourceEncoding;

public:
   CompilerIface( const CoreClass* cls );
   CompilerIface( const CoreClass* cls, const String &path );

   virtual ~CompilerIface();

   const ModuleLoader &loader() const { return m_loader; }
   ModuleLoader &loader() { return m_loader; }

   /** Returns a valid sequence instance if this object's user data is a "Falcon Sequence".

      Sequences can be used in sequential operations as the for-in loops,
      or in functional sequence operations (as map, filter and so on).

      Objects containing a Falcon Sequence as user data can declare
      this changing this function and returning the sequence data.
   */
   Sequence* getSequence() const { return m_bIsSequence ? static_cast<Sequence*>( m_user_data ) : 0; }

   /** Returns a valid sequence instance if this object's user data is a "Falcon Data".

      Sequences can be used in sequential operations as the for-in loops,
      or in functional sequence operations (as map, filter and so on).

      Objects containing a Falcon Sequence as user data can declare
      this changing this function and returning the sequence data.
   */
   FalconData *getFalconData() const { return m_bIsFalconData ? static_cast<FalconData*>( m_user_data ) : 0; }

   /** Return the inner user data that is attached to this item. */
   void *getUserData() const { return m_user_data; }

   /** Set a generic user data for this object.
      This user data is completely un-handled by this class; it's handling
      is completely demanded to user-defined sub-classes and/or to property-level
      reflection system.
   */
   void setUserData( void *data ) { m_user_data = data; }

   /** Set a FalconData as the user data for this object.
      FalconData class present a minimal interface that cooperates with this class:
      - It has a virtual destructor, that is called when the wrapping CoreObject instance is destroyed.
      - It provides a clone() method that is called when the wrapping CoreObject is cloned.
      - It provides a gcMark() method, called when this Object is marked.
      - Serialization support is available but defaulted to fail.
   */
   void setUserData( FalconData* fdata ) { m_bIsFalconData = true; m_user_data = fdata; }

   /** Set a Sequence as the user data for this object.
      Sequence class is derived from FalconData, and it adds an interface for serial
      access to items.
   */
   void setUserData( Sequence* sdata ) { m_bIsSequence = true; m_bIsFalconData = true; m_user_data = sdata; }

   /** Returns true if this object has the given class among its ancestors. */
   bool derivedFrom( const String &className ) const;


   /** Performs GC marking of the inner object data */
   virtual void gcMark( uint32 mk ) {}

   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;

   // we don't provide a clone() method
   virtual CoreObject *clone() const  { return 0; };
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
   virtual void gcMark( uint32 mk );

   // we don't provide a clone() method
};


}

}

#endif

/* end of compiler_mod.h */

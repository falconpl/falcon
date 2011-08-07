/*
   FALCON - The Falcon Programming Language.
   FILE: path.h

   Falcon core module -- Interface to Path.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_PATH_H
#define FALCON_CORE_PATH_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>

#include <falcon/usercarrier.h>
#include <falcon/path.h>

namespace Falcon {
namespace Ext {

/** We keep th path, the auth data and the query. */
class PathCarrier: public UserCarrier
{
public:
   Path m_path;
   
   String m_fulloc;
   String m_fullWinLoc;
   String m_winLoc;
   String m_winpath;
   
   PathCarrier( uint32 nprops ):
      UserCarrier(nprops)
   {}
   
   PathCarrier( const PathCarrier& other ):
      UserCarrier( other.itemCount() ),
      m_path( other.m_path )
   {}
   
   virtual ~PathCarrier()
   {      
   }
};


class ClassPath: public ClassUser
{
public:
   
   ClassPath();
   virtual ~ClassPath();

   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   
   //=============================================================
   //
   virtual void* createInstance( Item* params, int pcount ) const;   
   virtual void op_toString( VMContext* ctx, void* self ) const;
   
private:   
   
   //====================================================
   // Properties.
   //
   
   class PropertyResource: public PropertyString
   {
   public:
      PropertyResource( ClassUser* owner ):
         PropertyString( owner, "resource" )
      {}      
      virtual ~PropertyResource() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propRes;
   
   class PropertyLocation: public PropertyString
   {
   public:
      PropertyLocation( ClassUser* owner ):
         PropertyString( owner, "location" )
      {}      
      virtual ~PropertyLocation() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propLoc;
   
   class PropertyFullLocation: public PropertyString
   {
   public:
      PropertyFullLocation( ClassUser* owner ):
         PropertyString( owner, "fulloc" )
      {}      
      virtual ~PropertyFullLocation() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propFullLoc;
   
   class PropertyFile: public PropertyString
   {
   public:
      PropertyFile( ClassUser* owner ):
         PropertyString( owner, "file" )
      {}      
      virtual ~PropertyFile() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propFile;
   
   class PropertyExt: public PropertyString
   {
   public:
      PropertyExt( ClassUser* owner ):
         PropertyString( owner, "ext" )
      {}      
      virtual ~PropertyExt() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propExt;
   
   class PropertyFileExt: public PropertyString
   {
   public:
      PropertyFileExt( ClassUser* owner ):
         PropertyString( owner, "filext" )
      {}      
      virtual ~PropertyFileExt() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propFileExt;

   class PropertyEncoded: public PropertyString
   {
   public:
      PropertyEncoded( ClassUser* owner ):
         PropertyString( owner, "encoded" )
      {}      
      virtual ~PropertyEncoded() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propEncoded;

   
   class PropertyWLoc: public PropertyString
   {
   public:
      PropertyWLoc( ClassUser* owner ):
         PropertyString( owner, "wlocation" )
      {}      
      virtual ~PropertyWLoc() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propWLoc;
   
   class PropertyWFullLoc: public PropertyString
   {
   public:
      PropertyWFullLoc( ClassUser* owner ):
         PropertyString( owner, "wfulloc" )
      {}      
      virtual ~PropertyWFullLoc() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propFileWFullLoc;
   
   class PropertyWEncoded: public PropertyString
   {
   public:
      PropertyWEncoded( ClassUser* owner ):
         PropertyString( owner, "wencoded" )
      {}      
      virtual ~PropertyWEncoded() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propFileWEncoded;
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of path.h */

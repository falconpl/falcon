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
#include <falcon/classes/classuser.h>
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
      UserCarrier( other.dataSize() ),
      m_path( other.m_path )
   {}
   
   virtual ~PathCarrier()
   {
   }
   
   virtual PathCarrier* clone() const { return new PathCarrier(*this); }
};


class ClassPath: public ClassUser
{
public:
   
   ClassPath();
   virtual ~ClassPath();

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   
   //=============================================================
   //
   virtual void* createInstance( Item* params, int pcount ) const;   
   virtual void op_toString( VMContext* ctx, void* self ) const;
   
private:   
   
   //====================================================
   // Properties.
   //
   
   FALCON_DECLARE_STRPROPERTY( resource )
   FALCON_DECLARE_STRPROPERTY( location )
   FALCON_DECLARE_STRPROPERTY( fulloc )
   FALCON_DECLARE_STRPROPERTY( file )
   FALCON_DECLARE_STRPROPERTY( ext )
   FALCON_DECLARE_STRPROPERTY( filext )
   FALCON_DECLARE_STRPROPERTY( encoded )
   FALCON_DECLARE_STRPROPERTY( wlocation )
   FALCON_DECLARE_STRPROPERTY( wfulloc )
   FALCON_DECLARE_STRPROPERTY( wencoded )

   FALCON_DECLARE_METHOD( absolutize, "parent:[S]" );
   FALCON_DECLARE_METHOD( relativize, "parent:S" );
   FALCON_DECLARE_METHOD( canonicize, "" );
   FALCON_DECLARE_METHOD( cwd, "" );
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of path.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: deserializer.cpp

   Helper for cyclic joint structure deserialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Oct 2011 16:06:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/deserializer.h>
#include <falcon/class.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>

#include <vector>

namespace Falcon {

class Deserializer::MetaData
{
public:   
   /** Vector of numeric IDs on which an object depends. */
   typedef std::vector<size_t> IDVector;
   
   /** Data relative to a signle serialized item. */
   class ObjectData {
   public:
      void* m_data;
      size_t m_clsId;
      
      /** Items to be given back while deserializing. */
      IDVector m_deps;
      
      ObjectData():
         m_data(0),
         m_clsId(0)
      {}
      
      ObjectData( void* data,  size_t cls ):
         m_data(data),
         m_clsId(cls)
      {}
      
      ObjectData( const ObjectData& other ):
         m_data( other.m_data ),
         m_clsId( other.m_clsId )
      // ignore m_deps
      {}
   };
   
   typedef std::vector<ObjectData> ObjectDataVector;
   typedef std::vector<Class*> ClassVector;
   
   ClassVector m_clsVector;
   ObjectDataVector m_objVector;
   IDVector m_objBoundaries;
   
};

//===========================================================
//
//

Deserializer::Deserializer( ModSpace* ms ):
   m_modSpace( ms ),
   _meta(0),
   m_rd(0)
{  
}


Deserializer::~Deserializer()
{
   delete _meta;
}


void Deserializer::restore( DataReader* rd )
{
   delete _meta;
   _meta = new MetaData;   
   m_rd = rd;
   
   restoreClasses( rd );
}


void* Deserializer::next( Class&* handler )
{
   
}


bool Deserializer::hasNext() const
{
   
}


uint32 Deserializer::objCount() const
{
   
}


void restoreClasses( DataReader* rd )
{
   
}

}

/* end of deserializer.cpp */

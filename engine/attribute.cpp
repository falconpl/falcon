/*
   FALCON - The Falcon Programming Language.
   FILE: attribute.cpp

   Structure holding attributes for function, classes and modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 13:14:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/attribute.cpp"

#include <falcon/attribute.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/treestep.h>

namespace Falcon {

const char* Attribute::CLASS_NAME = "Attribute";

Attribute::Attribute():
         m_id(0),
         m_generator(0)
{}

Attribute::Attribute( uint32 id, const String& name, TreeStep* gen, const Item& dflt ):
   m_id(id),
   m_name(name),
   m_generator(gen),
   m_value(dflt)
{}

Attribute::Attribute( const Attribute& other ):
   m_id(other.m_id),
   m_name(other.m_name),
   m_generator(0)
{
   other.m_value.lock();
   m_value = other.m_value;
   other.m_value.unlock();

   if( other.m_generator != 0 ) {
      m_generator = other.m_generator->clone();
   }
}

Attribute::~Attribute()
{
   delete m_generator;
}


void Attribute::generator( TreeStep* gen )
{
   if( m_generator != 0 ) {
      delete m_generator;
   }
   m_generator = gen;
}


void Attribute::gcMark( uint32 mark )
{
   if ( m_name.currentMark() != mark ) {
      m_name.gcMark(mark);
      if( m_generator != 0 ) {
         m_generator->gcMark(mark);
      }
      m_value.gcMark(mark);
   }
}

void Attribute::store( DataWriter* stream ) const
{
   stream->write(m_id);
   stream->write(m_name);
}

void Attribute::restore( DataReader* stream )
{
   stream->read(m_id);
   stream->read(m_name);
}

void Attribute::flatten( ItemArray& subItems ) const
{
   subItems.append( m_value );

   if( m_generator != 0 )
   {
      subItems.append(Item( m_generator->handler(), m_generator ));
   }
}

void Attribute::unflatten( const ItemArray& subItems, uint32& pos )
{
   if( pos+2 <= subItems.length() )
   {
      // = is interlocked
      m_value = subItems[pos++];

      const Item& geni = subItems[pos++];
      if( ! geni.isNil() )
      {
         m_generator = static_cast<TreeStep*>( geni.asInst() ) ;
      }
   }
}

}

/* end of attribute.cpp */

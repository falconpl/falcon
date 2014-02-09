/*
   FALCON - The Falcon Programming Language.
   FILE: inspect.cpp

   Falcon core module -- inspect function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/inspect.cpp"

#include <falcon/cm/inspect.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>

#include <falcon/itemarray.h>
#include <falcon/itemdict.h>
#include <falcon/flexydict.h>
#include <falcon/falconinstance.h>
#include <falcon/falconclass.h>
#include <falcon/textwriter.h>

namespace Falcon {
namespace Ext {

Inspect::Inspect():
   Function( "inspect" )
{
   signature("X,[N],[N]");
   addParam("item");
   addParam("maxdepth");
   addParam("maxsize");
}

Inspect::~Inspect()
{
}

static void internal_inspect( TextWriter* tw, const Item& itm, int32 depth, int32 maxdepth, int32 maxsize )
{
   String temp;
   Class* cls = 0;
   void* data = 0;

   if( depth > 0 )
   {
      tw->write( PStep::renderPrefix(depth) );
   }
   else {
      depth = -depth;
   }

   if( maxdepth > 0 && depth == maxdepth )
   {
      tw->write("...");
      return;
   }

   // normally, inspect asks the describe function of the class to do the hard work,
   // but in case of classes, prototypes, arrays and dictionaries, it does a special work.
   switch( itm.type() )
   {
   case FLC_CLASS_ID_ARRAY:
   {
      ItemArray* ia = itm.asArray();
      if( ia->length() == 0 )
      {
         tw->write("[]");
      }
      else
      {
         tw->write("[\n");

         for( length_t pos = 0; pos < ia->length(); ++pos )
         {
            Item& item = ia->at(pos);
            internal_inspect(tw, item, depth+1, maxdepth, maxsize );
            if( pos+1 < ia->length() )
            {
               tw->write(",");
            }
            tw->write("\n");
         }

         tw->write( PStep::renderPrefix(depth ) );
         tw->write("]");
      }
   }
   break;

   case FLC_CLASS_ID_DICT:
   {
      ItemDict* id = itm.asDict();
      if( id->empty() == 0 )
      {
         tw->write("[=>]");
      }
      else
      {
         tw->write("[\n");

         class Rator: public ItemDict::Enumerator
         {
         public:
            Rator(TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize ):
               m_count(0),
               m_tw(tw), m_depth(depth), m_maxdepth(maxdepth), m_maxsize(maxsize)
               {}
            virtual ~Rator() {}

            virtual void operator()( const Item& key, Item& value )
            {
               if( m_count != 0 )
               {
                  m_tw->write(",\n");
               }
               m_count++;
               internal_inspect( m_tw, key, m_depth, m_maxdepth, m_maxsize );
               m_tw->write( " => ");
               internal_inspect( m_tw, value, m_depth, m_maxdepth, m_maxsize );
            }

         private:
            int32 m_count;
            TextWriter* m_tw;
            int32 m_depth;
            int32 m_maxdepth;
            int32 m_maxsize;
         };

         Rator rator(tw, depth+1, maxdepth, maxsize);
         id->enumerate(rator);
         tw->write( PStep::renderPrefix(depth ) );
         tw->write("\n]");
      }
   }
   break;

   case FLC_CLASS_ID_PROTO:
   {
      FlexyDict* fd = static_cast<FlexyDict*>(itm.asInst());

      if( fd->empty() == 0 )
      {
         tw->write("p{}");
      }
      else
      {
         tw->write("p{\n");

         class Rator: public Class::PVEnumerator
         {
         public:
            Rator(TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize ):
               m_tw(tw), m_depth(depth), m_maxdepth(maxdepth), m_maxsize(maxsize)
               {}
            virtual ~Rator() {}

            virtual void operator()( const String& property, Item& value )
            {
               m_tw->write( property );
               m_tw->write( " = ");
               internal_inspect( m_tw, value, -m_depth, m_maxdepth, m_maxsize );
               m_tw->write("\n");
            }

         private:
            TextWriter* m_tw;
            int32 m_depth;
            int32 m_maxdepth;
            int32 m_maxsize;
         };

         Rator rator(tw, depth+1, maxdepth, maxsize);
         fd->enumeratePV(rator);
         tw->write( PStep::renderPrefix(depth ) );
         tw->write("}");
      }
   }
   break;

   default:
      itm.forceClassInst(cls, data);

      if( cls->isFalconClass() )
      {
         class Rator: public Class::PropertyEnumerator {
         public:
            Rator(TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize, FalconInstance* fi, FalconClass* fcls ):
               m_tw(tw), m_depth(depth), m_maxdepth(maxdepth), m_maxsize(maxsize),
               m_fi(fi), m_fcls(fcls) {}

            virtual ~Rator() {}

            virtual bool operator()( const String& propName ) {
               Item item;
               m_tw->write( PStep::renderPrefix(m_depth) );
               const FalconClass::Property* prop = m_fcls->getProperty(propName);
               switch( prop->m_type )
               {
               case FalconClass::Property::t_static_prop:
                  m_tw->write( "static " );
                  /* no break */
               case FalconClass::Property::t_prop:
                  m_fi->getProperty( propName, item );
                  m_tw->write( propName );
                  m_tw->write( " = " );
                  internal_inspect(m_tw, item, -(m_depth+1), m_maxdepth, m_maxsize );
                  m_tw->write( "\n" );
                  break;

               case FalconClass::Property::t_inh:
                  m_tw->write( "from " );
                  m_fi->getProperty( propName, item );
                  internal_inspect(m_tw, item, -(m_depth+1), m_maxdepth, m_maxsize );
                  break;

               case FalconClass::Property::t_state:
                  m_tw->write( "[" );
                  m_tw->write( propName );
                  m_tw->write( "]\n" );
                  break;

               case FalconClass::Property::t_static_func:
                  m_tw->write( "static " );
                  /* no break */
               case FalconClass::Property::t_func:
                  m_tw->write( propName );
                  m_tw->write( "()\n" );
                  break;
               }
               return true;
            }

         public:
            TextWriter* m_tw; int32 m_depth; int32 m_maxdepth; int32 m_maxsize;
            FalconInstance* m_fi;
            FalconClass* m_fcls;
         };

         FalconInstance* fi = static_cast<FalconInstance*>( data );
         FalconClass* fcls = static_cast<FalconClass*>(cls);

         tw->write("Class ");
         tw->write( fcls->name() );
         tw->write("{\n");
         Rator rator( tw, depth+1, maxdepth, maxsize, fi, fcls );
         fcls->enumerateProperties(data, rator);
         tw->write( PStep::renderPrefix(depth) );
         tw->write("}");
      }
      else {
         cls->describe(data, temp, depth, maxsize );
         tw->write( temp );
      }
      break;
   }

   if( depth == 0 )
   {
      tw->write("\n");
   }
}

void Inspect::invoke( VMContext* ctx, int32 )
{
   Item* i_item = ctx->param(0);
   Item* i_maxdepth = ctx->param(1);
   Item* i_maxsize = ctx->param(2);

   if( i_item == 0
            || (i_maxdepth != 0 && !i_maxdepth->isOrdinal())
            || (i_maxsize != 0 && !i_maxsize->isOrdinal())
            )
   {
      throw paramError();
   }

   // prepare the local frame
   int64 maxdepth = i_maxdepth == 0 ? -1 : i_maxdepth->forceInteger();
   int64 maxsize = i_maxsize == 0 ? -1 : i_maxsize->forceInteger();

   internal_inspect( ctx->vm()->textOut(), *i_item, 0, maxdepth, maxsize );
   ctx->returnFrame();
}

}
}

/* end of inspect.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: inspector.cpp

   Helper class for dumping item contents.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Helper class for dumping item contents.

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/inspector.cpp"

#include <falcon/inspector.h>

#include <falcon/itemarray.h>
#include <falcon/itemdict.h>
#include <falcon/flexydict.h>
#include <falcon/falconinstance.h>
#include <falcon/falconclass.h>
#include <falcon/textwriter.h>

namespace Falcon {


Inspector::Inspector( TextWriter* tw ):
         m_tw(tw)
{
   tw->incref();
}


Inspector::~Inspector()
{
   m_tw->decref();
}


void Inspector::inspect( const Item& itm, int32 maxdepth, int32 maxsize )
{
   inspect_r(itm,0,maxdepth,maxsize);
}

void Inspector::inspect_r( const Item& itm, int32 depth, int32 maxdepth, int32 maxsize )
{
   String temp;
   Class* cls = 0;
   void* data = 0;

   if( depth > 0 )
   {
      m_tw->write( PStep::renderPrefix(depth) );
   }
   else {
      depth = -depth;
   }

   // normally, inspect asks the describe function of the class to do the hard work,
   // but in case of classes, prototypes, arrays and dictionaries, it does a special work.
   switch( itm.type() )
   {
   case FLC_ITEM_NIL:
      m_tw->write("Nil");
      break;

   case FLC_ITEM_BOOL:
      m_tw->write( itm.asBoolean() ? "true" : "false");
      break;

   case FLC_ITEM_INT:
      m_tw->write( "int(");
      temp.writeNumber( itm.asInteger() );
      m_tw->write(temp);
      m_tw->write( ")" );
      break;

   case FLC_ITEM_NUM:
      m_tw->write( "num(");
      temp.writeNumber( itm.asNumeric() );
      m_tw->write(temp);
      m_tw->write( ")" );
      break;

   case FLC_CLASS_ID_FUNC:
      m_tw->write( itm.asFunction()->name() );
      m_tw->write( "(" );
      m_tw->write( itm.asFunction()->getDescription() );
      m_tw->write( ")" );
      break;

   case FLC_ITEM_METHOD:
      m_tw->write( itm.asMethodClass()->name() );
      m_tw->write( "." );
      m_tw->write( itm.asMethodFunction()->name() );
      m_tw->write( "(" );
      m_tw->write( itm.asMethodFunction()->signature() );
      m_tw->write( ")" );
      break;

   case FLC_CLASS_ID_ARRAY:
   {
      ItemArray* ia = itm.asArray();
      if( ia->length() == 0 )
      {
         m_tw->write("[]");
      }
      else
      {
         m_tw->write("[\n");

         uint32 len = ia->length();
         for( length_t pos = 0; pos < len; ++pos )
         {
            if( pos > 0 )
            {
               m_tw->write(",\n");
            }

            Item& item = ia->at(pos);
            inspect_r( item, depth+1, maxdepth, maxsize );
         }
         m_tw->write("\n");
         m_tw->write( PStep::renderPrefix(depth) );
         m_tw->write("]");
      }
   }
   break;

   case FLC_CLASS_ID_DICT:
   {
      ItemDict* id = itm.asDict();
      if( id->empty() )
      {
         m_tw->write("[=>]");
      }
      else
      {
         class Rator: public ItemDict::Enumerator
         {
         public:
            Rator(Inspector* insp, TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize ):
               m_count(0),
               m_insp(insp),
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
               m_insp->inspect_r( key, m_depth, m_maxdepth, m_maxsize );
               m_tw->write( " => ");
               m_insp->inspect_r( value, -m_depth, m_maxdepth, m_maxsize );
            }

         private:
            int32 m_count;
            Inspector* m_insp;
            TextWriter* m_tw;
            int32 m_depth;
            int32 m_maxdepth;
            int32 m_maxsize;
         };

         if( maxdepth > 0 && depth >= maxdepth )
         {
            m_tw->write("[=>...]");
         }
         else {
            m_tw->write("[\n");
            Rator rator(this, m_tw, depth+1, maxdepth, maxsize);
            id->enumerate(rator);
            m_tw->write("\n");
            m_tw->write( PStep::renderPrefix(depth ) );
            m_tw->write("]");
         }
      }
   }
   break;

   case FLC_CLASS_ID_PROTO:
   {
      FlexyDict* fd = static_cast<FlexyDict*>(itm.asInst());

      if( fd->empty() )
      {
         m_tw->write("p{}");
      }
      else
      {
         class Rator: public Class::PVEnumerator
         {
         public:
            Rator(Inspector* insp, TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize ):
               m_insp(insp),
               m_tw(tw), m_depth(depth), m_maxdepth(maxdepth), m_maxsize(maxsize)
               {}
            virtual ~Rator() {}

            virtual void operator()( const String& property, Item& value )
            {
               m_tw->write( property );
               m_tw->write( " = ");
               m_insp->inspect_r( value, -m_depth, m_maxdepth, m_maxsize );
               m_tw->write("\n");
            }

         private:
            Inspector* m_insp;
            TextWriter* m_tw;
            int32 m_depth;
            int32 m_maxdepth;
            int32 m_maxsize;
         };

         if( maxdepth > 0 && depth >= maxdepth )
         {
            m_tw->write("p{...}");
         }
         else {
            Rator rator(this, m_tw, depth+1, maxdepth, maxsize);
            fd->enumeratePV(rator);
            m_tw->write("p{\n");
            m_tw->write( PStep::renderPrefix(depth ) );
            m_tw->write("}");
         }
      }
   }
   break;

   default:
      itm.forceClassInst(cls, data);

      if( cls->isFalconClass() )
      {
         class Rator: public Class::PropertyEnumerator {
         public:
            Rator(Inspector* insp, TextWriter* tw, int32 depth, int32 maxdepth, int32 maxsize, FalconInstance* fi, FalconClass* fcls ):
               m_insp(insp),
               m_tw(tw), m_depth(depth), m_maxdepth(maxdepth), m_maxsize(maxsize),
               m_fi(fi), m_fcls(fcls), count(0) {}

            virtual ~Rator() {}

            virtual bool operator()( const String& propName ) {
               Item item;
               m_tw->write( "\n" );
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
                  m_insp->inspect_r(item, -(m_depth), m_maxdepth, m_maxsize );
                  break;

               case FalconClass::Property::t_inh:
                  m_tw->write( "from " );
                  m_fi->getProperty( propName, item );
                  m_insp->inspect_r(item, -(m_depth), m_maxdepth, m_maxsize );
                  break;

               case FalconClass::Property::t_state:
                  m_tw->write( "[" );
                  m_tw->write( propName );
                  m_tw->write( "]" );
                  break;

               case FalconClass::Property::t_static_func:
                  m_tw->write( "static " );
                  /* no break */
               case FalconClass::Property::t_func:
                  m_tw->write( propName );
                  m_tw->write( "(" );
                  m_tw->write( prop->m_value.func->getDescription() );
                  m_tw->write( ")" );
                  break;
               }
               ++count;
               return true;
            }

         public:
            Inspector* m_insp;
            TextWriter* m_tw; int32 m_depth; int32 m_maxdepth; int32 m_maxsize;
            FalconInstance* m_fi;
            FalconClass* m_fcls;
            int32 count;
         };

         m_tw->write("Class ");
         m_tw->write( cls->name() );
         m_tw->write("{");

         if( maxdepth > 0 && depth >= maxdepth )
         {
            m_tw->write("...");
         }
         else {
            FalconInstance* fi = static_cast<FalconInstance*>( data );
            FalconClass* fcls = static_cast<FalconClass*>(cls);

            Rator rator( this, m_tw, depth+1, maxdepth, maxsize, fi, fcls );
            fcls->enumerateProperties(data, rator);

            m_tw->write( PStep::renderPrefix(depth) );
            if( rator.count != 0 ) { m_tw->write("\n"); }
         }

         m_tw->write("}");

      }
      else if (itm.type() == FLC_ITEM_USER)
      {
         class Rator: public Class::PVEnumerator
         {
         public:
            Rator(Inspector* insp, TextWriter* _tw, int32 d, int32 md, int32 ms ):
               m_insp(insp),
               tw(_tw), count(0), depth(d), maxdepth(md), maxsize(ms)
            {}

            virtual ~Rator() {}

            virtual void operator()( const String& property, Item& value )
            {
               tw->write("\n");
               tw->write( PStep::renderPrefix(depth) );
               tw->write(property);
               if( value.isMethod() || value.isFunction() )
               {
                  Function* func = value.isFunction() ? value.asFunction() : value.asMethodFunction();
                  tw->write("(");
                  tw->write( func->getDescription() );
                  tw->write(")");
               }
               else {
                  tw->write( " = " );
                  m_insp->inspect_r( value, -depth, maxdepth, maxsize );
               }
               ++count;
            }

         public:
            Inspector* m_insp;
            TextWriter* tw;
            int32 count;
            int32 depth;
            int32 maxdepth;
            int32 maxsize;
         }
         rator(this, m_tw, depth + 1, maxdepth, maxsize);

         m_tw->write("Class /*native*/ ");
         m_tw->write( cls->name() );
         m_tw->write("{");

         if( maxdepth > 0 && depth >= maxdepth )
         {
            m_tw->write("...");
         }
         else {
            cls->enumeratePV( data, rator );
            m_tw->write( PStep::renderPrefix(depth) );
            if( rator.count != 0 ) {
               m_tw->write("\n");
               m_tw->write( PStep::renderPrefix(depth) );
            }
         }
         m_tw->write("}");
      }
      else {
         cls->describe(data,temp,maxdepth-depth,maxsize);
         m_tw->write(temp);
      }
      break;
   }

   if( depth == 0 )
   {
      m_tw->write("\n");
   }
}
}

/* end of inspector.cpp */


/*
   FALCON - The Falcon Programming Language.
   FILE: exprcase.cpp

   Syntactic tree item definitions -- Case (switch branch selector)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Mar 2013 14:00:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#undef SRC
#define SRC "engine/psteps/exprcase.cpp"

#include "re2/re2.h"
#include <falcon/psteps/exprcase.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/symbol.h>
#include <falcon/gclock.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/range.h>
#include <falcon/stdhandlers.h>
#include <falcon/textwriter.h>

#include <falcon/stderrors.h>
#include <falcon/module.h>

#include <vector>

namespace Falcon {

class CaseEntry
{
public:
   typedef enum {
      e_t_none,
      e_t_nil,
      e_t_bool,
      e_t_int,
      e_t_int_range,
      e_t_string,
      e_t_string_range,
      e_t_regex,
      e_t_symbol,
      e_t_class
   }
   t_type;

   t_type m_type;
   typedef union {
      bool tof;

      struct {
         int64 int1;
         int64 int2;
      }
      ints;

      struct {
         String* string1;
         String* string2;
      }
      strings;

      struct {
         const Symbol* symbol;
         bool tilde;
      }
      sym;

      re2::RE2* regex;
   }
   t_data;

   t_data m_data;
   // We just store the class for convenience,
   // There is no reflection allowed for CaseEntry with classes,
   // and they are managed by the host expression (case).
   const Class* m_class;
   GCLock* m_lock;

   CaseEntry():
      m_type(e_t_none),
      m_class(0),
      m_lock(0)
   {}

   CaseEntry( const CaseEntry& other ):
        m_type(other.m_type),
        m_class(other.m_class),
        m_lock(0)
   {
     switch( m_type )
     {
     case e_t_none:
     case e_t_nil:
        /* do nothing */
        break;

     case e_t_bool:
        m_data.tof = other.m_data.tof;
        break;

     case e_t_int:
        m_data.ints.int1 = other.m_data.ints.int1;
        break;

     case e_t_int_range:
        m_data.ints.int1 = other.m_data.ints.int1;
        m_data.ints.int2 = other.m_data.ints.int2;
        break;

     case e_t_string:
        m_data.strings.string1 = new String(*other.m_data.strings.string1);
        m_data.strings.string1->bufferize();
        break;

     case e_t_string_range:
        m_data.strings.string1 = new String(*other.m_data.strings.string1);
        m_data.strings.string2 = new String(*other.m_data.strings.string2);
        m_data.strings.string1->bufferize();
        m_data.strings.string2->bufferize();
        break;

     case e_t_regex:
        m_data.regex = new re2::RE2(other.m_data.regex->pattern());
        break;

     case e_t_symbol:
        m_data.sym.symbol = other.m_data.sym.symbol;
        m_data.sym.tilde = other.m_data.sym.tilde;
        m_data.sym.symbol->incref();
        break;

     case e_t_class:
        if( m_class != 0 )
        {
           m_data.strings.string1 = new String( m_class->name() );
           m_lock = Engine::GC_lock(Item(m_class->handler(), const_cast<Class*>(m_class)));
        }
        else {
           m_data.strings.string1 = 0;
        }
        break;
     }
   }


   CaseEntry( t_type t, bool mode = false ):
      m_type(t),
      m_class(0),
      m_lock(0)
   {
      m_data.tof = mode;
   }


   CaseEntry( int64 value ):
      m_type(e_t_int),
      m_class(0),
      m_lock(0)
   {
      m_data.ints.int1 = value;
   }

   CaseEntry( int64 value1, int64 value2 ):
      m_type(e_t_int_range),
      m_class(0),
      m_lock(0)
   {
      m_data.ints.int1 = value1;
      m_data.ints.int2 = value2;
   }

   CaseEntry( String* str1 ):
      m_type(e_t_string),
      m_class(0),
      m_lock(0)
   {
      m_data.strings.string1 = new String(*str1);
      m_data.strings.string1->bufferize();
   }

   CaseEntry( String* str1, String* str2 ):
      m_type(e_t_string_range),
      m_class(0),
      m_lock(0)
   {
      m_data.strings.string1 = new String(*str1);
      m_data.strings.string2 = new String(*str2);
      m_data.strings.string1->bufferize();
      m_data.strings.string2->bufferize();
   }

   CaseEntry( re2::RE2* regex ):
      m_type(e_t_regex),
      m_class(0),
      m_lock(0)
   {
      m_data.regex = new re2::RE2(regex->pattern());
   }

   /** Warning: the symbol is not increffed here. */
   CaseEntry( const Symbol* symbol, bool hasTilde = false ):
      m_type(e_t_symbol),
      m_class(0),
      m_lock(0)
   {
      m_data.sym.symbol = symbol;
      m_data.sym.tilde = hasTilde;
   }

   /** Warning: the symbol is not increffed here. */
   CaseEntry( const Class* cls ):
      m_type(e_t_class),
      m_class(cls),
      m_lock(0)
   {
      m_data.strings.string1  = new String( cls->name() );
      m_lock = Engine::GC_lock(Item(cls->handler(), const_cast<Class*>(cls)));
   }


   ~CaseEntry()
   {
      clear();
   }

   void setNil()
   {
      clear();
      m_type = e_t_nil;
   }

   void setBool( bool bMode )
   {
      clear();
      m_type = e_t_bool;
      m_data.tof = bMode;
   }

   void setInt( int64 value )
   {
      clear();
      m_type = e_t_int;
      m_data.ints.int1 = value;
   }

   void setIntRange( int64 value1, int64 value2 )
   {
      clear();
      m_type = e_t_int_range;
      m_data.ints.int1 = value1;
      m_data.ints.int2 = value2;
   }

   void setString( String* str1 )
   {
      clear();
      m_type = e_t_string;
      m_data.strings.string1 = new String(*str1);
      m_data.strings.string1->bufferize();
   }

   void setStringRange( String* str1, String* str2 )
   {
      clear();
      m_type = e_t_string_range;
      m_data.strings.string1 = new String(*str1);
      m_data.strings.string2 = new String(*str2);
      m_data.strings.string1->bufferize();
      m_data.strings.string2->bufferize();
   }

   void setRegex( re2::RE2* regex )
   {
      clear();
      m_type = e_t_regex;
      m_data.regex = new re2::RE2(regex->pattern());
   }

   /** Warning: the symbol is not increffed here. */
   void setSymbol( const Symbol* symbol, bool hasTilde = false )
   {
      clear();
      m_type = e_t_symbol;
      m_data.sym.symbol = symbol;
      m_data.sym.tilde = hasTilde;
   }


   void setClass( const Class* cls )
   {
      clear();
      m_type = e_t_class;
      m_class = cls;
      m_data.strings.string1  = new String( cls->name() );
      m_lock = Engine::GC_lock(Item(cls->handler(), const_cast<Class*>(cls)));
   }

   void setForwardClass( const String& name )
   {
      clear(); // just in case
      m_type = e_t_class;
      m_class = 0;
      m_data.strings.string1 = new String( name );
   }

   void clear()
   {
      switch( m_type )
      {
      case e_t_string_range:
         delete m_data.strings.string2;
         /* no break */

      case e_t_string:
         delete m_data.strings.string1;
         break;

      case e_t_regex:
         delete m_data.regex;
         break;

      case e_t_symbol:
         m_data.sym.symbol->decref();
         break;

      case e_t_class:
         delete m_data.strings.string1;
         // already resolved?
         if( m_class != 0 && m_lock != 0 )
         {
            m_lock->dispose();
            m_lock = 0;
            m_class = 0;
         }
         break;

      default:
         /* do nothing */
         break;
      }

      m_type = e_t_none;
      // we don't hold anything of the class,
      // it's completely managed by the host entity (Case)
      m_class = 0;
   }


   void store( DataWriter* dw )
   {
      byte type = (byte) m_type;
      dw->write(type);

      switch( m_type )
      {
      case e_t_none:
      case e_t_nil:
         /* do nothing */
         break;

      case e_t_bool:
         dw->write( m_data.tof );
         break;

      case e_t_int:
         dw->write(m_data.ints.int1);
         break;

      case e_t_int_range:
         dw->write(m_data.ints.int1);
         dw->write(m_data.ints.int2);
         break;

      case e_t_string:
         dw->write(*m_data.strings.string1);
         break;

      case e_t_string_range:
         dw->write(*m_data.strings.string1);
         dw->write(*m_data.strings.string2);
         break;

      case e_t_regex:
         dw->write(m_data.regex->getPattern());
         break;

      case e_t_symbol:
         dw->write(m_data.sym.symbol->name());
         dw->write(m_data.sym.tilde);
         break;

      case e_t_class:
         dw->write(*m_data.strings.string1);
         // classes are flattened by the host statement.
         break;

      }
   }

   void restore( DataReader* dr )
   {
      String temp, temp2;
      byte type = 0;

      // just in case...
      clear();

      dr->read(type);
      //we'll assign the type is everything is ok.
      // **notice** that each read operation might throw

      switch( (t_type) type )
      {
      case e_t_none:
         /* do nothing -- after clear, we're already e_t_none */
         return;

      case e_t_nil:
         m_type = e_t_nil;
         return;

      case e_t_bool:
         dr->read(m_data.tof);
         m_type = e_t_bool;
         return;

      case e_t_int:
         dr->read(m_data.ints.int1);
         m_type = e_t_int;
         return;

      case e_t_int_range:
         dr->read(m_data.ints.int1);
         dr->read(m_data.ints.int2);
         m_type = e_t_int_range;
         return;

      case e_t_string:
         dr->read(temp);
         m_data.strings.string1 = new String(temp); // we know it's bufferized
         m_type = e_t_string;
         return;

      case e_t_string_range:
         dr->read(temp);
         dr->read(temp2);
         // we didn't throw, so it's ok.

         m_data.strings.string1 = new String(temp); // we know it's bufferized
         m_data.strings.string2 = new String(temp2); // we know it's bufferized
         m_type = e_t_string_range;
         return;

      case e_t_regex:
         dr->read(temp);
         {
            re2::RE2* regex = new re2::RE2(temp);
            if( regex->error_code() != 0 )
            {
               Error* error = new IOError(ErrorParam(e_deser, __LINE__, SRC )
                        .extra(String("Invalid regex (").N(regex->error_code()).A(") ").A(regex->error().c_str()) )
                      );
               delete regex;
               throw error;
            }
            m_data.regex = regex;
            m_type = e_t_regex;
         }
         return;

      case e_t_symbol:
         dr->read(temp);
         // read the tilde status now,
         dr->read(m_data.sym.tilde);
         m_data.sym.symbol = Engine::getSymbol(temp);
         m_type = e_t_symbol;
         return;

      case e_t_class:
         m_data.strings.string1 = 0;
         m_class = 0; // class will be resolved later.
         dr->read(temp);
         m_data.strings.string1 = new String(temp);
         m_type = e_t_class;
         return;
      }

      // invalid enumeration!
      Error* error = new IOError(ErrorParam(e_deser, __LINE__, SRC )
                     .extra(String("Invalid ExprCase entry type (").N(type).A(")") )
                   );
            throw error;
   }


   bool verify( const Item& value ) const
   {
      switch( m_type )
      {
      case e_t_none:
         return false;

      case e_t_nil:
         return value.isNil();

      case e_t_bool:
         return value.isBoolean() ? m_data.tof == value.asBoolean() : false;

      case e_t_int:
         if( value.isInteger() )
         {
            return m_data.ints.int1 == value.asInteger();
         }
         else if( value.isNumeric() )
         {
            return m_data.ints.int1 == (int64) value.asNumeric();
         }
         return false;

      case e_t_int_range:
         if( value.isInteger() )
         {
            int64 v = value.asInteger();
            return m_data.ints.int1 <= v && v <= m_data.ints.int2;
         }
         else if( value.isNumeric() )
         {
            int64 v = (int64) value.asNumeric();
            return m_data.ints.int1 <= v && v <= m_data.ints.int2;
         }
         return false;

      case e_t_string:
         if( value.isString() )
         {
            return *value.asString() == *m_data.strings.string1;
         }
         return false;


      case e_t_string_range:
         if( value.isString() )
         {
            const String& str = *value.asString();
            return *m_data.strings.string1 <= str && str <= *m_data.strings.string2;
         }
         return false;

      case e_t_regex:
        if( value.isString() )
        {
           const String& str = *value.asString();
           return RE2::PartialMatchN( str, *m_data.regex, 0, 0);
        }
        return false;

      // symbol values are checked separately by user classes.
      // here we check symbols by name.
      case e_t_symbol:
         if( value.type() == FLC_CLASS_ID_SYMBOL )
         {
            return static_cast<Symbol*>(value.asInst())->name() == m_data.sym.symbol->name();
         }
         return false;


      // In case of unresolved classes, we check the string/class by name.
      // otherwise, the block is verified if the incoming item has a class
      // derived from the one we have in this entry.
      case e_t_class:
         if( m_class != 0 )
         {
            Class* cls = 0;
            void* data = 0;
            value.forceClassInst(cls, data);
            return cls->isDerivedFrom(m_class);
         }
         else if( value.type() == FLC_CLASS_ID_STRING )
         {
            return *value.asString() == *m_data.strings.string1;
         }
         else if( value.type() == FLC_CLASS_ID_CLASS )
         {
            Class* cls = static_cast<Class*>(value.asInst());
            return cls->name() == *m_data.strings.string1;
         }

         return false;
      }

      return false;
   }

};


class ExprCase::Private
{
public:
   typedef std::vector<CaseEntry*> EntryList;
   EntryList m_entries;

   Private() {}
   Private( const Private& other )
   {
      EntryList::const_iterator iter = other.m_entries.begin();
      while( iter != other.m_entries.end() )
      {
         CaseEntry* oe = *iter;
         m_entries.push_back(new CaseEntry(*oe));
         ++iter;
      }
   }

   ~Private() {
      EntryList::iterator iter = m_entries.begin();
      while( iter != m_entries.end() )
      {
         delete * iter;
         ++iter;
      }
   }
};


ExprCase::ExprCase( int line, int chr ):
   Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_case )
   m_trait = e_trait_case;
   apply = apply_;
   _p = new Private;
}
ExprCase::ExprCase( const ExprCase& other ):
   Expression( other )
{
   apply = apply_;
   _p = new Private(*other._p);
}


ExprCase::~ExprCase()
{
   delete _p;
}


void ExprCase::render( TextWriter* tw, int32 depth ) const
{
   String temp1, temp2;
   tw->write( renderPrefix(depth) );
   //tw->write( "case ");  -- this is written by our hosts

   Private::EntryList::iterator iter = _p->m_entries.begin();

   while( iter != _p->m_entries.end() )
   {
      if( iter != _p->m_entries.begin() )
      {
         tw->write(", ");
      }

      CaseEntry *ce = *iter;
      switch( ce->m_type )
      {
      case CaseEntry::e_t_none: tw->write("/* Blank CaseEntry */"); break;
      case CaseEntry::e_t_nil: tw->write("nil"); break;
      case CaseEntry::e_t_bool: tw->write( ce->m_data.tof ? "true" : "false" ); break;
      case CaseEntry::e_t_int: tw->write( String("").N(ce->m_data.ints.int1) ); break;
      case CaseEntry::e_t_int_range: tw->write( String("").N(ce->m_data.ints.int1) + " to " + String("").N(ce->m_data.ints.int2) ); break;
      case CaseEntry::e_t_string:
         ce->m_data.strings.string1->escapeFull(temp1);
         tw->write( "\"" );
         tw->write(temp1);
         tw->write( "\"" );
         break;

      case CaseEntry::e_t_string_range:
         ce->m_data.strings.string1->escapeFull(temp1);
         ce->m_data.strings.string2->escapeFull(temp2);
         tw->write( "\"" );
         tw->write(temp1);
         tw->write( "\" to \"" );
         tw->write(temp2);
         tw->write( "\"" );
         break;

      case CaseEntry::e_t_regex:
         temp1.fromUTF8( ce->m_data.regex->pattern().c_str() );
         tw->write("r\"");
         tw->write(temp1);
         tw->write("\"");
         break;

      case CaseEntry::e_t_symbol:
         if(ce->m_data.sym.tilde)
         {
            tw->write("~");
         }
         tw->write(ce->m_data.sym.symbol->name());
         break;

      case CaseEntry::e_t_class:
         tw->write(*ce->m_data.strings.string1);
         break;
      }

      ++iter;
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


bool ExprCase::simplify( Item& ) const
{
   return false;
}


void ExprCase::apply_( const PStep*, VMContext* ctx )
{
   // not really used.
   ctx->popCode();
   ctx->pushData(Item());
}



static bool internal_setValue( CaseEntry* entry, const Item& value )
{
   switch( value.type() )
   {
   case FLC_ITEM_INT:
      entry->setInt(value.asInteger());
      break;

   case FLC_ITEM_NUM:
      entry->setInt((int64)value.asNumeric());
      break;

   case FLC_CLASS_ID_RANGE:
      {
         Range* r = static_cast<Range*>( value.asInst() );
         entry->setIntRange(r->start(), r->end());
      }
      break;

   case FLC_CLASS_ID_STRING:
      entry->setString( value.asString() );
      break;

   case FLC_CLASS_ID_ARRAY:
      {
         ItemArray& ia = *value.asArray();
         if( ia.length() != 2 || ! ia[0].isString() || ! ia[1].isString() )
         {
            return false;
         }

         entry->setStringRange(ia[0].asString(), ia[1].asString() );
      }
      break;

   case FLC_CLASS_ID_RE:
      entry->setRegex( static_cast<re2::RE2*>(value.asInst()) );
      break;

   case FLC_CLASS_ID_SYMBOL:
      {
         const Symbol* sym = value.asSymbol();
         entry->setSymbol(sym);
      }
      break;

   default:
      return false;
   }

   return true;
}


void ExprCase::addNilEntry()
{
   _p->m_entries.push_back( new CaseEntry(CaseEntry::e_t_nil) );
}

void ExprCase::addBoolEntry( bool tof )
{
   _p->m_entries.push_back( new CaseEntry(CaseEntry::e_t_bool, tof) );
}

void ExprCase::addEntry( int64 value )
{
   _p->m_entries.push_back(new CaseEntry(value));
}

void ExprCase::addEntry( int64 value1, int64 value2 )
{
   _p->m_entries.push_back(new CaseEntry(value1, value2));
}

void ExprCase::addEntry( const String& str1 )
{
   _p->m_entries.push_back(new CaseEntry(new String(str1)));
}

void ExprCase::addEntry( const String& str1, const String& str2 )
{
   _p->m_entries.push_back(new CaseEntry(new String(str1),new String(str2)));
}

void ExprCase::addEntry( re2::RE2* regex )
{
   _p->m_entries.push_back(new CaseEntry(regex));
}

void ExprCase::addEntry( const Symbol* symbol, bool hasTilde )
{
   _p->m_entries.push_back(new CaseEntry(symbol, hasTilde));
}

void ExprCase::addEntry( Class* cls )
{
   _p->m_entries.push_back(new CaseEntry(cls));
}

bool ExprCase::addEntry( const Item& value )
{
   CaseEntry* entry = new CaseEntry;
   bool res = internal_setValue( entry, value );
   if( ! res )
   {
      delete entry;
   }
   else {
      _p->m_entries.push_back(entry);
   }
   return res;
}

bool ExprCase::verify( const Item& value ) const
{
   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
            iter !=  _p->m_entries.end();
            ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->verify(value ) )
      {
         return true;
      }
   }

   return false;
}


bool ExprCase::verifySymbol( const Symbol* value ) const
{
   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
            iter !=  _p->m_entries.end();
            ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->m_type == CaseEntry::e_t_symbol && entry->m_data.sym.symbol == value )
      {
         return true;
      }
   }

   return false;
}

void ExprCase::store( DataWriter* dw )
{
   uint32 size = _p->m_entries.size();
   dw->write(size);

   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
              iter !=  _p->m_entries.end();
              ++iter )
     {
        CaseEntry* entry = *iter;
        entry->store(dw);
     }

}


void ExprCase::restore( DataReader* dr )
{
   uint32 size = 0;
   dr->read( size );

   for ( uint32 i = 0; i < size; ++i )
   {
     CaseEntry* entry = new CaseEntry;
     try
     {
        entry->restore(dr);
        _p->m_entries.push_back(entry);
     }
     catch( ... )
     {
        delete entry;
        throw;
     }

   }

}


void ExprCase::flatten( ItemArray& subItems )
{
   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
                 iter !=  _p->m_entries.end();
                 ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->m_class != 0 )
      {
         subItems.append(Item(entry->m_class->handler(), const_cast<Class*>(entry->m_class)) );
      }
   }
}


void ExprCase::unflatten( ItemArray& subItems )
{
   static Collector* coll = Engine::collector();
   uint32 count = 0;

   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
                 iter !=  _p->m_entries.end();
                 ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->m_type == CaseEntry::e_t_class )
      {
         if( count == subItems.length() )
         {
            throw new IOError( ErrorParam(e_deser, __LINE__, SRC )
                     .extra("Deserialized classes not matching in case"));
         }
         Class* cls = static_cast<Class*>(subItems[count].asInst());
         entry->m_class = cls;
         entry->m_lock = coll->lock(subItems[count]);
         entry->m_data.strings.string1 = new String(cls->name());
         ++count;
      }
   }

   if( count != subItems.length() )
   {
      throw new IOError( ErrorParam(e_deser, __LINE__, SRC )
               .extra("Deserialized classes not matching after case"));
   }
}



bool ExprCase::verifyItem( const Item& item, VMContext* ctx ) const
{
   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
                 iter != _p->m_entries.end();
                 ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->verify(item) )
      {
         return true;
      }

      if( entry->m_type == CaseEntry::e_t_symbol )
      {
         Item* var = ctx->resolveSymbol(entry->m_data.sym.symbol, false);
         if ( var != 0 && var->exactlyEqual(item) )
         {
            return true;
         }
      }
   }

   return false;
}


bool ExprCase::verifyType( const Item& item, VMContext* ctx ) const
{
   int64 type = (int64) item.type();

   for ( Private::EntryList::iterator iter = _p->m_entries.begin();
                    iter != _p->m_entries.end();
                    ++iter )
   {
      CaseEntry* entry = *iter;
      if( entry->m_type == CaseEntry::e_t_symbol )
      {
         // Resolve NOW
         const Symbol* sym = entry->m_data.sym.symbol;
         Item* value = ctx->resolveSymbol( sym, false );
         if( value == 0 ) {
            throw FALCON_SIGN_XERROR(LinkError, e_undef_sym, .extra(sym->name()).line(line()) );
         }
         else if( value->isInteger() )
         {
            entry->m_type = CaseEntry::e_t_int;
            entry->m_data.ints.int1 = value->asInteger();
            sym->decref();
         }
         else if( value->isClass() )
         {
            entry->m_type = CaseEntry::e_t_class;
            entry->m_class = static_cast<Class*>(value->asInst());
            entry->m_data.strings.string1 = new String(entry->m_class->name());
            sym->decref();
         }
         else {
            throw FALCON_SIGN_XERROR(LinkError, e_inv_const_val, .extra("Must be a type number or a class").line(line()) );
         }
      }

      if( entry->m_type == CaseEntry::e_t_int )
      {
         if( entry->m_data.ints.int1 == type )
         {
            return true;
         }
      }
      else if( entry->m_type == CaseEntry::e_t_class && entry->m_class != 0)
      {
         Class* itmClass = 0;
         void* itmValue = 0;
         item.forceClassInst(itmClass, itmValue);
         if( itmClass->isDerivedFrom(entry->m_class) )
         {
            return true;
         }
      }
   }

   return false;
}


int32 ExprCase::arity() const
{
   return (int32) _p->m_entries.size();
}

TreeStep* ExprCase::nth( int32 n ) const
{
   static Class* clsRange = Engine::instance()->stdHandlers()->rangeClass();
   static Class* clsSym = Engine::instance()->stdHandlers()->symbolClass();
   static Class* clsRegex = Engine::instance()->stdHandlers()->reClass();

   if( n < 0 ) n = _p->m_entries.size() + n;
   if( n < 0 || n >= (int32) _p->m_entries.size() )
   {
      return 0;
   }

   CaseEntry* ce = _p->m_entries[n];

   ExprValue* result = 0;
   switch(ce->m_type)
   {
   case CaseEntry::e_t_none:
   case CaseEntry::e_t_nil:
      result = new ExprValue;
      break;

   case CaseEntry::e_t_bool: result = new ExprValue( Item().setBoolean(ce->m_data.tof)); break;
   case CaseEntry::e_t_int: result = new ExprValue( Item().setBoolean(ce->m_data.ints.int1 != 0)); break;
   case CaseEntry::e_t_int_range: result = new ExprValue( FALCON_GC_STORE( clsRange, new Range(ce->m_data.ints.int1, ce->m_data.ints.int2) ) ); break;
   case CaseEntry::e_t_string: result = new ExprValue( FALCON_GC_HANDLE(new String( *ce->m_data.strings.string1) ) ); break;
   case CaseEntry::e_t_regex: result = new ExprValue( Item( clsRegex, ce->m_data.regex ) ); break;
   case CaseEntry::e_t_symbol: result = new ExprValue( Item(clsSym, ce->m_data.sym.symbol) ); break;
   case CaseEntry::e_t_class:
      if( ce->m_class != 0 )
      {
         result = new ExprValue( Item(ce->m_class->handler(), const_cast<Class*>(ce->m_class) ) );
      }
      else {
         result = new ExprValue( FALCON_GC_HANDLE(new String( *ce->m_data.strings.string1) ) );
      }
      break;

   case CaseEntry::e_t_string_range:
      {
         String* str1 = new String( *ce->m_data.strings.string1 );
         String* str2 = new String( *ce->m_data.strings.string2 );
         ItemArray* arr = new ItemArray();
         arr->append(FALCON_GC_HANDLE(str1));
         arr->append(FALCON_GC_HANDLE(str2));
         result = new ExprValue( FALCON_GC_HANDLE(arr) );
      }
      break;
   }

   return result;
}


bool ExprCase::setNth( int32 n, TreeStep* ts )
{
   if( n < 0 ) n = _p->m_entries.size() + n;
   if( n == (int32) _p->m_entries.size() )
   {
      return append(ts);
   }

   if( n < 0 || n >= (int32) _p->m_entries.size()
            || ts->category() != TreeStep::e_cat_expression
            || static_cast<Expression*>(ts)->trait() != Expression::e_trait_value
            || ts->parent() != 0)
   {
      return false;
   }

   // we just need the value in the expression.
   Item& value = static_cast<ExprValue*>(ts)->item();

   return internal_setValue( _p->m_entries[n], value );
}


bool ExprCase::insert( int32 n, TreeStep* ts )
{
   if( n < 0 ) n = _p->m_entries.size() + n;
   if( n == (int32) _p->m_entries.size() )
   {
      return append(ts);
   }

   if( n < 0 || n >= (int32) _p->m_entries.size()
            || ts->category() != TreeStep::e_cat_expression
            || static_cast<Expression*>(ts)->trait() != Expression::e_trait_value
            || ts->parent() != 0)
   {
      return false;
   }

   // we just need the value in the expression.
   Item& value = static_cast<ExprValue*>(ts)->item();
   CaseEntry* ce = new CaseEntry();
   if( internal_setValue( ce, value ) )
   {
      _p->m_entries.insert(_p->m_entries.begin() + n, ce );
      return true;
   }
   else {
      delete ce;
      return false;
   }
}


bool ExprCase::append( TreeStep* ts )
{
   if( ts->category() != TreeStep::e_cat_expression
      || static_cast<Expression*>(ts)->trait() != Expression::e_trait_value
      || ts->parent() != 0)
   {
      return false;
   }

   // we just need the value in the expression.
   Item& value = static_cast<ExprValue*>(ts)->item();

   CaseEntry* ce = new CaseEntry();
   if( internal_setValue( ce, value ) )
   {
      _p->m_entries.push_back( ce );
      return true;
   }
   else {
      delete ce;
      return false;
   }
}


bool ExprCase::remove( int32 n )
{
   if( n < 0 ) n = _p->m_entries.size() + n;
   if( n >= (int)_p->m_entries.size() )
   {
      return false;
   }

   delete _p->m_entries[n];
   _p->m_entries.erase(_p->m_entries.begin() + n);
   return true;
}


}

/* end of exprcase.cpp */

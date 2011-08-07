/*
   FALCON - The Falcon Programming Language.
   FILE: property.cpp

   Abstraction for properties and reflected properties.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/property.cpp"

#include <falcon/property.h>
#include <falcon/item.h>
#include <falcon/classuser.h>
#include <falcon/paramerror.h>

namespace Falcon {


Property::Property( ClassUser* uc, const String &name, bool bCarried ):
   m_name(name),
   m_owner(uc),
   m_bCarried(bCarried)
{
   uc->add(this);
}


Property::~Property()
{
}


void Property::checkType( bool ok, const String& required )
{
   if( ! ok )
   {
      throw new ParamError( ErrorParam(e_prop_invalid, __LINE__, SRC )
         .extra(name() +"=" + required) 
         );
   }
}

//===================================================================
// PropertyCarried
//

PropertyCarried::PropertyCarried( ClassUser* uc, const String &name ):
   Property( uc, name, true )
{
}

//===================================================================
// PropertyString
//

void PropertyString::get( void* instance, Item& target )
{
   UserCarrier* uric = static_cast<UserCarrier*>(instance);
   Item* cache = carried( uric );
   
   String* str;
   if( cache->isNil() )
   {
      str = new String;
      cache->setString(str);
   }
   else
   {
      str = cache->asString();
   }
   
   const String& value = getString( instance );
   if( value != *str )
   {
      str->bufferize( value );
   }
   
   // str is in cache.
   target.copy(*cache);   
}

//===================================================================
// PropertyData
//

void PropertyData::set( void* instance, const Item& value )
{
   UserCarrier* uric = static_cast<UserCarrier*>(instance);
   Item* cache = carried( uric );
      
   if( cache->isNil() )
   {
      initCacheItem( *cache );
      fassert( cache->isUser() );
      
      // ensure the item gets marked.
      cache->garbage();
   }
   
   Class* cls, *vcls;
   void* udata, *vudata;
   cache->forceClassInst( cls, udata );
   value.forceClassInst( vcls, vudata );
   if( cls != vcls && (cls->typeID() != vcls->typeID() || cls->name() != cls->name()) )
   {
      throw new ParamError( ErrorParam(e_prop_invalid, __LINE__, SRC )
         .extra(name() +"=" + cls->name()) 
         );
   }
   
   //we can proceed
   update( vudata, *cache );
}

void PropertyData::get( void* instance, Item& target )
{
   UserCarrier* uric = static_cast<UserCarrier*>(instance);
   Item* cache = carried( uric );   
   
   if( cache->isNil() )
   {
      initCacheItem( *cache );
      fassert( cache->isUser() );
      
      // ensure the item gets marked.
      cache->garbage();
   }
   
   fetch( cache->asInst() );  
   target.copy(*cache);   
}


//===================================================================
// PropertyReflect
//


PropertyReflect::PropertyReflect( ClassUser* uc, const String &name ):
   PropertyCarried( uc, name ),
   m_displacement( not_displaced ),
   m_offset(0),
   m_max_size(0),
   m_type( e_rt_none )
{
}


void PropertyReflect::get( void* instance, Item& value )
{
   void *displaced = instance;
   
   // first, get the displacement.
   if( m_displacement != not_displaced )
   {
      char* disp = (char*) instance;
      disp+= m_displacement;
      displaced = *((void**)disp);
   }
   
   char* inst = (char*) displaced;
   inst += m_offset;
   
   switch( m_type )
   {
      case e_rt_none:
         // nothing to reflect...
         break;
         
      case e_rt_bool:
         value.setBoolean( *(bool*)inst );
         break;
         
      case e_rt_char:
         value.setInteger( (int64)*(char*)inst );
         break;
         
      case e_rt_uchar:
         value.setInteger( (int64)*(unsigned char*)inst );
         break;
         
      case e_rt_short:
         value.setInteger( (int64)*(short*)inst );
         break;
         
      case e_rt_ushort:
         value.setInteger( (int64)*(unsigned short*)inst );
         break;
         
      case e_rt_int:
         value.setInteger( (int64)*(int*)inst );
         break;
         
      case e_rt_uint:
         value.setInteger( (int64)*(unsigned int*)inst );
         break;
         
      case e_rt_long:
         value.setInteger( (int64)*(long*)inst );
         break;
         
      case e_rt_ulong:         
         value.setInteger( (int64)*(unsigned long*)inst );
         break;
         
      case e_rt_int64:
         value.setBoolean( *(int64*)inst );
         break;
         
      case e_rt_uint64:         
         value.setBoolean( (int64) *(uint64*)inst );
         break;
         
      case e_rt_float:
         value.setNumeric( (numeric) *(float*)inst );
         break;
         
      case e_rt_double:
         value.setNumeric( (numeric) *(float*)inst );
         break;
         
      case e_rt_ldouble:         
         value.setNumeric( (numeric) *(float*)inst );
         break;
         
      case e_rt_charp:
      case e_rt_wcharp:
      case e_rt_string:
         // strings are carried
         {
            UserCarrier* uc = (UserCarrier*) instance;
            Item* cache = uc->dataAt( carrierPos() );
            String* str;
            if( cache->isNil() )
            {
               str = new String;
               cache->setString( str ); // will also garbage
               cache->copied();  // so that we have copy on write downstream
            }
            else
            {
               str = cache->asString();
            }
            
            switch( m_type ) // yes, again
            {
               case e_rt_charp: str->fromUTF8( (char*) inst ); break;
               case e_rt_wcharp: str->bufferize( (wchar_t*) inst ); break;
               default: str->bufferize( *(String*) inst ); break;
            }
            
            // finally, store the cache.
            value = *cache;
         }         
         break;
         
         
      case e_rt_buffer:
         // TODO
         break;                  
   }
}


void PropertyReflect::set( void* instance, const Item& value )
{
   void *displaced = instance;
   
   // first, get the displacement.
   if( m_displacement != not_displaced )
   {
      char* disp = (char*) instance;
      disp+= m_displacement;
      displaced = *((void**)disp);
   }
   
   char* inst = (char*) displaced;
   inst += m_offset;
   
   const char* ptype = "";
   
   switch( m_type )
   {
      case e_rt_none:
         // nothing to reflect...
         break;
         
      case e_rt_bool:
         *(bool*)inst = value.isTrue();
         break;
         
      case e_rt_char:
         if( value.isOrdinal() )
         {
            *(char*)inst = (char) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_uchar:
         if( value.isOrdinal() )
         {
            *(unsigned char*)inst = (unsigned char) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_short:
         if( value.isOrdinal() )
         {
            *(short*)inst = (short) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_ushort:
         if( value.isOrdinal() )
         {
            *(unsigned short*)inst = (unsigned short) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_int:
         if( value.isOrdinal() )
         {
            *(int*)inst = (int) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_uint:
         if( value.isOrdinal() )
         {
            *(unsigned int*)inst = (unsigned int) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_long:
         if( value.isOrdinal() )
         {
            *(long*)inst = (long) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_ulong:         
         if( value.isOrdinal() )
         {
            *(unsigned long*)inst = (unsigned long) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_int64:
         if( value.isOrdinal() )
         {
            *(int64*)inst = (int64) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_uint64:         
         if( value.isOrdinal() )
         {
            *(uint64*)inst = (uint64) value.forceInteger();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_float:
         if( value.isOrdinal() )
         {
            *(float*)inst = (float) value.forceNumeric();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_double:
         if( value.isOrdinal() )
         {
            *(double*)inst = (double) value.forceNumeric();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_ldouble:         
         if( value.isOrdinal() )
         {
            *(long double*)inst = (long double) value.forceNumeric();
            return;
         }
         // naaa, we can't set it.
         ptype = "N";
         break;
         
      case e_rt_charp:
         // We don't care about updating the cache.
         if( value.isString() )
         {
            value.asString()->toCString( (char*) inst, m_max_size );
            return;
         }
         // naaa, we can't set it.
         ptype = "S";
         break;
         
      case e_rt_wcharp:
         // We don't care about updating the cache.
         if( value.isString() )
         {
            value.asString()->toWideString( (wchar_t*) inst, m_max_size );
            return;
         }
         // naaa, we can't set it.
         ptype = "S";
         break;
         
      case e_rt_string:
         // We don't care about updating the cache.
         if( value.isString() )
         {
            *(String*) inst = *value.asString();
            return;
         }         
         // naaa, we can't set it.
         ptype = "S";
         break;
         
         
      case e_rt_buffer:
         // TODO
         break;                  
   }
   
   // signal a type error.
   throw new ParamError( ErrorParam(e_prop_invalid, __LINE__, SRC )
      .extra(name()+"=" + ptype)
      );
}


void PropertyReflect::reflect( void* base, bool& value )
{
   m_type = e_rt_bool;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, char& value )
{
   m_type = e_rt_char;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, unsigned char& value )
{
   m_type = e_rt_uchar;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, short& value )
{
   m_type = e_rt_short;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, unsigned short& value )
{
   m_type = e_rt_ushort;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, int& value )
{
   m_type = e_rt_int;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, unsigned int& value )
{
   m_type = e_rt_uint;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, long& value )
{
   m_type = e_rt_long;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, unsigned long& value )
{
   m_type = e_rt_ulong;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, int64& value )
{
   m_type = e_rt_int64;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, uint64& value )
{
   m_type = e_rt_uint64;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, float& value )
{
   m_type = e_rt_float;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, double& value )
{
   m_type = e_rt_double;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, long double& value )
{
   m_type = e_rt_ldouble;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::reflect( void* base, char*& value, uint32 max_size, bool bIsBuffer )
{
   m_type = bIsBuffer ? e_rt_buffer : e_rt_charp;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
   m_max_size = max_size;
}


void PropertyReflect::reflect( void* base, wchar_t*& value, uint32 max_size )
{
   m_type = e_rt_wcharp;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
   m_max_size = max_size;
}


void PropertyReflect::reflect( void* base, String& value )
{
   m_type = e_rt_string;
   m_offset = (uint32) (((uint64)base) - ((uint64)&value));
}


void PropertyReflect::displace( void* base, void*& displacePtr )
{
   m_displacement = (uint32) (((uint64)base) - ((uint64)displacePtr));
}
   

//===================================================================
// PropertyReflectRO
//


PropertyReflectRO::PropertyReflectRO( ClassUser* uc, const String &name ):
   PropertyReflect( uc, name )
{}

void PropertyReflectRO::set( void*, const Item& )
{
   // signal a type error.
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC )
      .extra(name())
      );
}


//===================================================================
// PropertyReflectWO
//

PropertyReflectWO::PropertyReflectWO( ClassUser* uc, const String &name ):
   PropertyReflect( uc, name )
{}

void PropertyReflectWO::get( void*, Item& )
{
   // signal a type error.
   throw new ParamError( ErrorParam(e_prop_wo, __LINE__, SRC )
      .extra(name())
      );
}

}

/* end of property.cpp */

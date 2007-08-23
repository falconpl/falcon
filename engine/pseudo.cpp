/*
   FALCON - The Falcon Programming Language.
   FILE: pseudo.cpp
   $Id: pseudo.cpp,v 1.4 2007/06/09 07:39:37 jonnymind Exp $

   Assembly pseudo code variable management
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio ago 25 2005
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
   Assembly pseudo code variable management
*/

#include <falcon/common.h>
#include <falcon/symbol.h>
#include <falcon/stream.h>
#include <fasm/pseudo.h>

namespace Falcon {

Pseudo::Pseudo( int l, type_t opt, const char *str, bool disp ):
   m_line( l ),
   m_type( opt ),
   m_disp( disp ),
   m_fixed( false )
{
   m_value.dispstring = new String( str );
}


Pseudo::~Pseudo()
{
   if ( m_type == tdispstring )
   {
      delete m_value.dispstring;
   }
   else if ( m_type == tswitch_list )
   {
      ListElement *iter = m_value.child->begin();
      while( iter != 0 ) {
         Pseudo *child = (Pseudo *) iter->data();
         delete child;
         iter = iter->next();
      }
      delete  m_value.child ;
   }
}

void Pseudo::write( Stream *out ) const
{
   switch( type() )
   {
      case Pseudo::imm_int:
      {
         if ( fixed() ) {
            int32 iPos = endianInt32( (int32) asInt() );
            out->write( &iPos, sizeof( iPos ) );
         }
         else {
            int64 iPos = endianInt64( asInt() );
            out->write( &iPos, sizeof( iPos ) );
         }
      }
      break;

      case Pseudo::imm_range:
      {
         int32 iPos = endianInt32( asRangeStart() );
         out->write( &iPos, sizeof( iPos ) );
         iPos = endianInt32( asRangeEnd() );
         out->write( &iPos, sizeof( iPos ) );
      }
      break;

      case Pseudo::imm_double:
      {
         double dNum = endianNum( asDouble() );
         out->write( &dNum, sizeof( dNum ) );
      }
      break;

      case Pseudo::imm_string:
      {
         int iData = endianInt32( asString().id() );
         out->write( &iData, sizeof( iData ) );
      }
      break;

      case Pseudo::tsymbol:
      {
         Symbol *sym = asSymbol();
         uint32 iData = fixed() ? endianInt32( sym->id() ) : endianInt32( sym->itemId() ) ;
         out->write( &iData, sizeof( iData ) );
      }
      break;

      case Pseudo::tname:
      {
         asLabel()->write( out );
      }
      break;

      // in any other case, it can't write anything.
   }
}

bool  Pseudo::operator <( const Pseudo &other ) const
{
   if ( ((int32) type()) < ((int32) other.type()) )
      return true;
   else if ( type() == other.type() )
      switch( type() )
      {
         case tnil: return false;
         case imm_int: return asInt() < other.asInt();
         case imm_double: return asDouble() < other.asDouble();
         case imm_string: return asString() < other.asString();
         case imm_range: return asRangeStart() < other.asRangeStart();
         case tsymbol: return asSymbol()->id() < other.asSymbol()->id();
      }

   return false;
}

int PseudoPtrTraits::compare( const void *firstArea, const void *secondv ) const
{
   Pseudo *first = *(Pseudo **) firstArea;
   Pseudo *second = (Pseudo *) secondv;

   if ( ((int32) first->type()) < ((int32) second->type()) )
      return -1;
   else if ( first->type() == second->type() )
   {
      switch( first->type() )
      {
         case Pseudo::tnil: return 0;

         case Pseudo::imm_int:
            if( first->asInt() < second->asInt() )
               return -1;
            else if( first->asInt() > second->asInt() )
               return 1;
            return 0;

         case Pseudo::imm_double:
            if( first->asDouble() < second->asDouble() )
               return -1;
            else if( first->asDouble() > second->asDouble() )
               return 1;
            return 0;

         case Pseudo::imm_string: return first->asString().compare( second->asString() );
         case Pseudo::imm_range:
             if( first->asRangeStart() < second->asRangeStart() )
               return -1;
            else if( first->asRangeStart() > second->asRangeStart() )
               return 1;
            return 0;

         case Pseudo::tsymbol:
            if( first->asSymbol()->id() < second->asSymbol()->id() )
               return -1;
            else if( first->asSymbol()->id() > second->asSymbol()->id() )
               return 1;
            return 0;
      }
   }

   return 1;
}


void PseudoPtrTraits::destroy( void *item ) const
{
   Pseudo *pseudo = *(Pseudo **) item;
   if( pseudo->disposeable() )
      delete pseudo;
}

bool PseudoPtrTraits::owning() const
{
   return true;
}

namespace traits {
   PseudoPtrTraits t_pseudoptr;
}

}

/* end of pseudo.cpp */

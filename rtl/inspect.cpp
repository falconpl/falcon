/*
   FALCON - The Falcon Programming Language.
   FILE: print.cpp
   $Id: inspect.cpp,v 1.15 2007/08/11 10:22:36 jonnymind Exp $

   Basic module
   -------------------------------------------------------------------
   Author: $AUTHOR
   Begin: $DATE
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/attribute.h>
#include <falcon/stream.h>

namespace Falcon { namespace Ext {

void inspect_internal( VMachine *vm, const Item *elem, int32 level, bool add = true );

void inspect_internal( VMachine *vm, const Item *elem, int32 level, bool add )
{
   uint32 count;
   int32 i;
   bool addline = true;
   Stream *stream = vm->stdErr();
   if ( stream == 0 )
   {
      stream = vm->stdOut();
      if ( stream == 0 )
         return;
   }

   if( level < 0 ) {
      level = -level;
      addline = false;
   }

   if ( add )
      for ( i = 0; i < level*3; i ++ )
      {
         stream->put( 0x20 ); // space
      }

   if ( elem == 0 ) {
      stream->writeString( "Nothing" );
      if ( addline )
         stream->writeString( "\n" );
      return;
   }

   String temp;

   switch( elem->type() )
   {
     case FLC_ITEM_NIL:
         stream->writeString( "Nil" );
      break;

      case FLC_ITEM_INT:
         temp.writeNumber( elem->asInteger() );
         stream->writeString( "int(" );
         stream->writeString( temp );
         stream->writeString( ")" );
      break;

      case FLC_ITEM_NUM:
         temp.writeNumber( elem->asNumeric(), "%g" );
         stream->writeString( "num(" );
         stream->writeString( temp );
         stream->writeString( ")" );
      break;

      case FLC_ITEM_RANGE:
         temp = "[";
         temp.writeNumber( (int64) elem->asRangeStart() );
         temp += ":";
         if( ! elem->asRangeIsOpen() ) {
            temp.writeNumber( (int64) elem->asRangeEnd() );
         }
         temp += "]";
         stream->writeString( temp );
      break;

      case FLC_ITEM_STRING:
         stream->writeString( "\"" );
         stream->writeString( *elem->asString() );
         stream->writeString( "\"" );
      break;

      case FLC_ITEM_ATTRIBUTE:
         stream->writeString( "{attrib:" );
         stream->writeString( elem->asAttribute()->name() );
         stream->writeString( "}" );
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = elem->asArray();
         temp = "Array[";
         temp.writeNumber( (int64) arr->length() );
         temp += "]{\n";
         stream->writeString( temp );

         for( count = 0; count < arr->length() ; count++ ) {
            inspect_internal( vm, & ((*arr)[count]), level + 1 );
         }

         for ( i = 0; i < level; i ++ )
         {
            stream->writeString( "   " );
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = elem->asDict();
         temp = "Dict[";
         temp.writeNumber( (int64) dict->length() );
         temp += "]{\n";
         stream->writeString( temp );

         Item key, value;
         dict->traverseBegin();
         while( dict->traverseNext( key, value ) )
         {
            inspect_internal( vm, &key, -(level + 1) );
            stream->writeString( " => " );
            inspect_internal( vm, &value, level + 1, false );
         }
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *arr = elem->asObject();
         stream->writeString( "Object of class " + arr->instanceOf()->name() );
         stream->writeString( " {\n" );

         for( count = 0; count < arr->propCount() ; count++ ) {
            for ( i = 0; i < (level+1); i ++ )
            {
               stream->writeString("   ");
            }
            stream->writeString( arr->getPropertyName( count ) + " => " );
            inspect_internal( vm, &arr->getPropertyAt(count), level + 1, false );
         }
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_CLASS:
         stream->writeString( "Class " + elem->asClass()->symbol()->name() );
      break;

      case FLC_ITEM_METHOD:
      {
         if ( ! elem->asModule()->isAlive() )
         {
            stream->writeString( "Dead method" );
         }
         else
         {
            temp = "Method 0x";
            temp.writeNumberHex( (uint64) elem->asMethodObject() );
            temp += "->" + elem->asMethodFunction()->name();
            stream->writeString( temp );

            Item itemp;
            itemp.setObject( elem->asMethodObject() );
            inspect_internal( vm, &itemp, level + 1, true );
            itemp.setFunction( elem->asMethodFunction(), elem->asModule() );
            inspect_internal( vm, &itemp, level + 1, true );
            for ( i = 0; i < level; i ++ )
            {
               stream->writeString("   ");
            }
            stream->writeString( "}" );
         }
      }
      break;

      case FLC_ITEM_FBOM:
         temp = "Fbom Method id=";
         temp.writeNumber( (int64) elem->getFbomMethod() );
         stream->writeString( temp );
         stream->writeString( " on " );
         {
            Item other;
            elem->getFbomItem( other );
            inspect_internal( vm, &other, level, false );
         }

      break;

      case FLC_ITEM_CLSMETHOD:
         temp = "Cls.Method 0x";
         temp.writeNumberHex( (uint64) elem->asMethodObject() );
         temp += "->" + elem->asMethodClass()->symbol()->name();
         stream->writeString( temp );
      break;

      case FLC_ITEM_FUNC:
      {
         if ( ! elem->asModule()->isAlive() )
         {
            stream->writeString( "Dead function" );
         }
         else {
            Symbol *funcSym = elem->asFunction();

            if ( funcSym->isExtFunc() )
            {
               stream->writeString( "Ext. Function " + funcSym->name() );
            }
            else {
               stream->writeString( "Function " + funcSym->name() );

               FuncDef *def = funcSym->getFuncDef();
               uint32 itemId = def->onceItemId();
               if ( itemId != FuncDef::NO_STATE )
               {
                  if ( elem->asModule()->globals().itemAt( itemId ).isNil() )
                     stream->writeString( "{ not called }");
                  else
                     stream->writeString( "{ called }");
               }
            }
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         stream->writeString( "Ref to " );
         inspect_internal( vm, elem->dereference(), level, false );
      break;

      default:
         stream->writeString( "Invalid type");
   }
   if ( addline )
      stream->writeString( "\n" );

   stream->flush();
}

FALCON_FUNC  inspect ( ::Falcon::VMachine *vm )
{
   for( int i = 0; i < vm->paramCount(); i ++ )
      inspect_internal( vm, vm->param(i), 0 );
}


}}

/* end of print.cpp */

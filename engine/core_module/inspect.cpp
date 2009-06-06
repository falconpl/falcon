/*
   FALCON - The Falcon Programming Language.
   FILE: print.cpp

   Basic module
   -------------------------------------------------------------------
   Author: $AUTHOR
   Begin: $DATE

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/corefunc.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>

namespace Falcon {
namespace core {

void inspect_internal( VMachine *vm, const Item *elem, int32 level, int32 maxLevel, int32 maxSize, bool add = true, bool addLine=true );

void inspect_internal( VMachine *vm, const Item *elem, int32 level, int32 maxLevel, int32 maxSize, bool add, bool addline )
{

   uint32 count;
   int32 i;
   Stream *stream = vm->stdErr();

   // return if we reached the maximum level.
   if ( maxLevel >= 0 && level > maxLevel )
   {
      stream->writeString( "..." );
      if ( addline )
         stream->writeString( "\n" );
      return;
   }

   if ( stream == 0 )
   {
      stream = vm->stdOut();
      if ( stream == 0 )
         return;
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

      case FLC_ITEM_BOOL:
         stream->writeString( elem->asBoolean() ? "true" : "false" );
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
         elem->toString(temp);
         stream->writeString( temp );
      break;

      case FLC_ITEM_STRING:
         stream->writeString( "\"" );
         if ( maxSize < 0 || elem->asString()->length() < (uint32) maxSize )
         {
            stream->writeString( *elem->asString() );
            stream->writeString( "\"" );
         }
         else {
            stream->writeString( elem->asString()->subString(0, maxSize ) );
            stream->writeString( " ... \"" );
         }
      break;

      case FLC_ITEM_LBIND:
         stream->writeString( "&" );
         stream->writeString( *elem->asLBind() );
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = elem->asMemBuf();
         temp = "MemBuf(";
         temp.writeNumber( (int64) mb->length() );
         temp += ",";
         temp.writeNumber( (int64) mb->wordSize() );
         temp += ")";

         if ( maxSize == 0 )
            stream->writeString( temp );
         else {
            temp += " [\n";

            String fmt;
            int limit = 0;
            switch ( mb->wordSize() )
            {
               case 1: fmt = "%02X"; limit = 24; break;
               case 2: fmt = "%04X"; limit = 12; break;
               case 3: fmt = "%06X"; limit = 9; break;
               case 4: fmt = "%08X"; limit = 6; break;
            }

            int written = 0;
            uint32 max = maxSize < 0 || mb->length() < (uint32) maxSize ? mb->length() : (uint32) maxSize;
            for( count = 0; count < max; count++ )
            {
               temp.writeNumber( (int64)  mb->get( count ), fmt );
               temp += " ";
               written ++;
               if ( written == limit )
               {
                  temp += "\n";
                  written = 0;
               }
               stream->writeString( temp );
               temp = "";
            }
            if ( count == (uint32) maxSize )
               stream->writeString( " ... " );
            stream->writeString( "]" );
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = elem->asArray();
         temp = "Array[";
         temp.writeNumber( (int64) arr->length() );
         temp += "]";
         stream->writeString( temp );

         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( "{\n" );

         for( count = 0; count < arr->length(); count++ ) {
            inspect_internal( vm, & ((*arr)[count]), level + 1, maxLevel, maxSize, true, true );
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
         temp += "]";
         stream->writeString( temp );

         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( "{\n" );

         Item key, value;
         dict->traverseBegin();
         while( dict->traverseNext( key, value ) )
         {
            inspect_internal( vm, &key, level + 1, maxLevel, maxSize, true, false );
            stream->writeString( " => " );
            inspect_internal( vm, &value, level + 1, maxLevel, maxSize, false, true );
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
         CoreObject *arr = elem->asObjectSafe();
         stream->writeString( "Object of class " + arr->generator()->symbol()->name() );
         if ( level == maxLevel )
         {
            stream->writeString( "{...}" );
            break;
         }

         stream->writeString( " {\n" );
         const PropertyTable &pt = arr->generator()->properties();

         for( count = 0; count < pt.added() ; count++ )
         {
            for ( i = 0; i < (level+1); i ++ )
            {
               stream->writeString("   ");
            }
            const String &propName = *pt.getKey( count );
            stream->writeString( propName + " => " );
            Item dummy;
            arr->getProperty( propName, dummy);
            inspect_internal( vm, &dummy, level + 1, maxLevel, maxSize, false, true );
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
         temp = "Method ";
         temp += "->" + elem->asMethodFunc()->symbol()->name();
         stream->writeString( temp );

         Item itemp;
         elem->getMethodItem( itemp );

         inspect_internal( vm, &itemp, level + 1, maxLevel, maxSize, true, true );
         for ( i = 0; i < level; i ++ )
         {
            stream->writeString("   ");
         }
         stream->writeString( "}" );
      }
      break;

      case FLC_ITEM_CLSMETHOD:
         temp = "Cls.Method 0x";
         temp.writeNumberHex( (uint64) elem->asMethodClassOwner() );
         temp += "->" + elem->asMethodClass()->symbol()->name();
         stream->writeString( temp );
      break;

      case FLC_ITEM_FUNC:
      {
         const Symbol *funcSym = elem->asFunction()->symbol();

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
               if ( elem->asFunction()->liveModule()->globals().itemAt( itemId ).isNil() )
                  stream->writeString( "{ not called }");
               else
                  stream->writeString( "{ called }");
            }
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         stream->writeString( "Ref to " );
         inspect_internal( vm, elem->dereference(), level + 1, maxLevel, maxSize, false, true );
      break;

      default:
         stream->writeString( "Invalid type");
   }
   if ( addline )
      stream->writeString( "\n" );

   stream->flush();
}


/*#
   @function inspect
   @inset core_basic_io
   @param item The item to be inspected.
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @brief Displays the deep contents of an item.

   This is mainly a debugging function that prints all the available
   informations on the item on the auxiliary stream. This function
   should not be used except for testing scripts and checking what
   they put in arrays, dictionaries, objects, classes or simple items.

   Output is sent to the VM auxiliary stream; for stand-alone scripts,
   this translates into the "standard error stream". Embedders may provide
   simple debugging facilities by overloading and interceptiong the VM
   auxiliary stream and provide separate output for that.

   This function traverse arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a maxlen parameter.

   You may create personalized inspect functions using forward bindings, like
   the following:
   @code
   compactInspect = .[inspect depth|1 maxLen|15]
   @endcode

   And then, you may inspect a list of item with something like:
   @code
   linsp = .[ dolist _compactInspect x ]
   linsp( ["hello", "world"] )
   @endcode

*/

FALCON_FUNC  inspect ( ::Falcon::VMachine *vm )
{
   Item *i_item = vm->param(0);
   Item *i_depth = vm->param(1);
   Item *i_maxLen = vm->param(2);

   if ( i_item == 0
      || ( i_depth != 0 && ! i_depth->isNil() && ! i_depth->isOrdinal() )
      || ( i_maxLen != 0 && ! i_maxLen->isNil() && ! i_maxLen->isOrdinal() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,[N],[N]") ) );
      return;
   }

   int32 depth = (int32) (i_depth == 0 || i_depth->isNil() ? 3 : i_depth->forceInteger());
   int32 maxlen = (int32) (i_maxLen == 0 || i_maxLen->isNil() ? 60 : i_maxLen->forceInteger());

   inspect_internal( vm, i_item, 0, depth, maxlen );
}


}}

/* end of print.cpp */

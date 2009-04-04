/*
   FALCON - The Falcon Programming Language.
   FILE: corefunc.cpp

   Abstract live function object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Jan 2009 14:54:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Abstract live function object.
*/

#include <falcon/corefunc.h>
#include <falcon/vm.h>
#include <falcon/eng_messages.h>

namespace Falcon
{

void CoreFunc::readyFrame( VMachine *vm, uint32 paramCount )
{
   // eventually check for named parameters
   if ( vm->regBind().flags() == 0xF0 )
   {
      const SymbolTable *symtab;

      if( m_symbol->isFunction() )
         symtab = &m_symbol->getFuncDef()->symtab();
      else
         symtab = m_symbol->getExtFuncDef()->parameters();

      vm->regBind().flags(0);
      // We know we have (probably) a named parameter.
      uint32 size = vm->m_stack->size();
      uint32 paramBase = size - paramCount;
      ItemVector iv(8);

      uint32 pid = 0;

      // first step; identify future binds and pack parameters.
      while( paramBase+pid < size )
      {
         Item &item = vm->m_stack->itemAt( paramBase+pid );
         if ( item.isFutureBind() )
         {
            // we must move the parameter into the right position
            iv.push( &item );
            for( uint32 pos = paramBase + pid + 1; pos < size; pos ++ )
            {
               vm->m_stack->itemAt( pos - 1 ) = vm->m_stack->itemAt( pos );
            }
            vm->m_stack->itemAt( size-1 ).setNil();
            size--;
            paramCount--;
         }
         else
            pid++;
      }
      vm->m_stack->resize( size );

      // second step: apply future binds.
      for( uint32 i = 0; i < iv.size(); i ++ )
      {
         Item &item = iv.itemAt( i );

         // try to find the parameter
         const String *pname = item.asLBind();
         Symbol *param = symtab == 0 ? 0 : symtab->findByName( *pname );
         if ( param == 0 ) {
            throw new CodeError( ErrorParam( e_undef_param, __LINE__ ).extra(*pname) );
         }

         // place it in the stack; if the stack is not big enough, resize it.
         if ( vm->m_stack->size() <= param->itemId() + paramBase )
         {
            paramCount = param->itemId()+1;
            vm->m_stack->resize( paramCount + paramBase );
         }

         vm->m_stack->itemAt( param->itemId() + paramBase ) = item.asFBind()->origin();
      }
   }

   // ensure against optional parameters.
   if( m_symbol->isFunction() )
   {
      FuncDef *tg_def = m_symbol->getFuncDef();

      if( paramCount < tg_def->params() )
      {
         vm->m_stack->resize( vm->m_stack->size() + tg_def->params() - paramCount );
         paramCount = tg_def->params();
      }

      vm->createFrame( paramCount );

      // now we can change the stack base
      vm->m_stackBase = vm->m_stack->size();

      // space for locals
      if ( tg_def->locals() > 0 )
         vm->m_stack->resize( vm->m_stackBase + tg_def->locals() );

      vm->m_code = tg_def->code();
      vm->m_currentModule = m_lm;
      vm->m_currentGlobals = &m_lm->globals();
      vm->m_symbol = m_symbol;

      //jump
      vm->m_pc_next = 0;

      // If the function is not called internally by the VM, another run is issued.
      /*
      if( callMode == e_callNormal || callMode == e_callInst )
      {
         // hitting the stack limit forces the RET code to raise a return event,
         // and this forces the machine to exit run().
         frame->m_break = true;
         run();
      }
      */
   }
   else
   {
      vm->createFrame( paramCount );

      // now we can change the stack base
      vm->m_stackBase = vm->m_stack->size();

      vm->m_symbol = m_symbol; // so we can have adequate tracebacks.
      vm->m_currentModule = m_lm;
      vm->m_currentGlobals = &m_lm->globals();

      vm->m_pc_next = VMachine::i_pc_call_external;
   }
}

}

/* end of corefunc.cpp */

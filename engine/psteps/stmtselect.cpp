/*
   FALCON - The Falcon Programming Language.
   FILE: stmtselect.cpp

   Syntactic tree item definitions -- select statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/select.cpp"

#include <falcon/expression.h>
#include <falcon/pcode.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/module.h>

#include <falcon/errors/linkerror.h>

#include <falcon/psteps/stmtselect.h>

#include <map>
#include <deque>

namespace Falcon {

class StmtSelect::Private
{
public:
   // blocks are kept separate for easier destruction and accounting.
   typedef std::deque<SynTree*> BlockList;
   BlockList m_blocks;
   
   // blocks ordered for int
   typedef std::map<int64, SynTree*> IntBlocks;
   
   // blocks ordered for class -- used only in case of perfect matches.
   typedef std::map<Class*, SynTree*> ClassBlocks;
  
   // List of requested classes, and also declaration-ordered list of classes.
   typedef std::deque< StmtSelect::SelectRequirement* > ClassList;
   
   IntBlocks m_intBlocks;
   ClassBlocks m_classBlocks;
   ClassList m_classList;
   
   
   Private() {}
   ~Private() 
   {
      BlockList::iterator iter = m_blocks.begin();
      while( iter != m_blocks.end() )
      {
         delete *iter;
         ++iter;
      }
      
      ClassList::iterator cliter = m_classList.begin();
      while( cliter != m_classList.end() )
      {
         delete *cliter;
         ++cliter;
      }
   }
};
   

StmtSelect::StmtSelect( Expression* expr, int32 line, int32 chr ):
   Statement( e_stmt_select, line, chr ),
   _p( new Private ),
   m_expr( expr ),
   m_defaultBlock(0),
   m_module(0)
{
   if (expr != 0 )
   {
      apply = apply_;
   }
   else
   {
      // we're used just as a dictionary.
      apply = 0;
   }
}

StmtSelect::~StmtSelect() 
{
   delete m_expr;
   delete m_defaultBlock;
   delete _p;
}


void StmtSelect::describeTo( String& tgt ) const
{
   tgt = "select " + m_expr->oneLiner() +"\n";
   // TODO 
}


void StmtSelect::oneLinerTo( String& tgt ) const
{
   tgt = "select " + m_expr->oneLiner();
}

bool StmtSelect::addSelectType( int64 typeId, SynTree* block )
{
   // refuse to add if already existing.
   if ( _p->m_intBlocks.find( typeId ) != _p->m_intBlocks.end() )
   {
      return false;
   }
   
   // save the block only if not just pushed in the last operation.
   if( _p->m_blocks.empty() || _p->m_blocks.back() != block )
   {
      _p->m_blocks.push_back( block );
   }
   
   // anyhow, associate the request.
   _p->m_intBlocks[typeId ] = block;
   return true;
}

bool StmtSelect::addSelectClass( Class* cls, SynTree* block )
{
   // refuse to add if already existing.
   if ( _p->m_classBlocks.find( cls ) != _p->m_classBlocks.end() )
   {
      return false;
   }
   
   // save the block only if not just pushed in the last operation.
   if( _p->m_blocks.empty() || _p->m_blocks.back() != block )
   {
      _p->m_blocks.push_back( block );
   }
   
   // anyhow, associate the request.
   _p->m_classBlocks[ cls ] = block;
   
   // record the requirement -- as already resolved.
   SelectRequirement* r = new SelectRequirement( cls->name(), block, this );
   r->m_cls = cls;
   _p->m_classList.push_back( r );
   
   return true;
}

Requirement* StmtSelect::addSelectName( const String& name, SynTree* block )
{      
   if( _p->m_blocks.empty() || _p->m_blocks.back() != block )
   {
      _p->m_blocks.push_back( block );
   }
   
   SelectRequirement* req = new SelectRequirement( name, block, this );  
   _p->m_classList.push_back( req );
   return req;
}


SynTree* StmtSelect::findBlockForType( int64 typeId ) const
{
   Private::IntBlocks::iterator pos = _p->m_intBlocks.find( typeId );
   if( pos != _p->m_intBlocks.end() )
   {
      return pos->second;
   }
   return 0;
}


SynTree* StmtSelect::findBlockForClass( Class* cls ) const
{
   Private::ClassBlocks::iterator pos = _p->m_classBlocks.find( cls );
   if( pos != _p->m_classBlocks.end() )
   {
      return pos->second;
   }
   
   // the class wasn't found; but we may have a predecessor in our list.   
   Private::ClassList::iterator iter = _p->m_classList.begin();
   while( iter != _p->m_classList.end() )
   {
      Class* base = (*iter)->m_cls;
      if( base != 0 && cls->isDerivedFrom( base ) )
      {
         return (*iter)->m_block;
      }
      ++iter;
   }
   
   // no luck
   return 0;
}
   

SynTree* StmtSelect::findBlockForItem( const Item& itm ) const
{
   // fist try with the types.
   if( _p->m_intBlocks.size() > 0 )
   {
      // The type is that for the item...
      int64 tid = itm.dereference()->type();
      if( tid == FLC_ITEM_USER )
      {
         // ... or specified by the class.
         tid = itm.asClass()->typeID();
      }
      
      Private::IntBlocks::iterator pos = _p->m_intBlocks.find( tid );
      if( pos != _p->m_intBlocks.end() )
      {
         return pos->second;
      }
   }
   
   // no luck with integer representing type ids -- try to find a class.
   Class* cls;
   void* data;
   itm.forceClassInst( cls, data );
   SynTree* res = findBlockForClass( cls );
   if( res != 0 )
   {
      return res;
   }
   
   // the only hope left is the default block
   return m_defaultBlock;
}
 

bool StmtSelect::setDefault( SynTree* block )
{
   if( m_defaultBlock )
   {
      return false;
   }
   
   m_defaultBlock = block;
   return true;
}


void StmtSelect::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtSelect* self = static_cast<const StmtSelect*>(ps);
   
   CodeFrame& cf = ctx->currentCode();
   // first time around? -- call the expression.
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1; 
      if( ctx->stepInYield( self->m_expr, cf ) ) 
      {
         return;
      }
   }
   
   SynTree* res = self->findBlockForItem( ctx->topData() );
   
   // we're gone
   ctx->popCode();
   // and so is the topdata.
   ctx->popData();
   
   // but if the syntree wants to do something...
   if( res != 0 )
   {
      ctx->pushCode( res );
   }
}

//================================================================
// The requirer
//
   
void StmtSelect::SelectRequirement::onResolved( 
         const Module* source, const Symbol* sym, Module*, Symbol*  )
{
   Item* itm = sym->defaultValue();
   if( itm == 0 || (!itm->isOrdinal()&& ! itm->isClass()) )
   {
      throw new LinkError( ErrorParam( m_owner->m_expr == 0 ? e_catch_invtype : e_select_invtype )
         .line( m_block->sr().line() )
         .module( m_owner->m_module != 0 ? m_owner->m_module->uri() : "" )
         .origin( ErrorParam::e_orig_linker )
         .symbol( sym->name() )
         .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
         );
   }
   
   // an integer? 
   if( itm->isOrdinal() )
   {
      int64 tid = itm->forceInteger();
      if( m_owner->_p->m_intBlocks.find( tid ) != m_owner->_p->m_intBlocks.end() )
      {
         throw new LinkError( ErrorParam( m_owner->m_expr == 0 ? e_catch_clash : e_switch_clash )
            .line( m_block->sr().line() )
            .module( m_owner->m_module != 0 ? m_owner->m_module->uri() : "" )
            .origin( ErrorParam::e_orig_linker )
            .symbol( sym->name() )
            .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
            );
      }
      m_owner->_p->m_intBlocks[ tid ] = m_block;
   }
   else
   {
      fassert( itm->asClass()->isMetaClass() );
      Class* cls = static_cast<Class*>(itm->asInst());
      if( m_owner->_p->m_classBlocks.find( cls ) != m_owner->_p->m_classBlocks.end() )
      {
         throw new LinkError( ErrorParam( m_owner->m_expr == 0 ? e_catch_clash : e_switch_clash )
            .line( m_block->sr().line() )
            .module( m_owner->m_module != 0 ? m_owner->m_module->uri() : "" )
            .origin( ErrorParam::e_orig_linker )
            .symbol( sym->name() )
            .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
            );
      }
      m_cls = cls;
      m_owner->_p->m_classBlocks[ cls ] = m_block;
   }
}

}

/* end of stmtselect.cpp */

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
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/module.h>

#include <falcon/errors/linkerror.h>

#include <falcon/psteps/stmtselect.h>
#include <falcon/classes/classrequirement.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/itemarray.h>

#include <map>
#include <deque>
#include <vector>

namespace Falcon {

class StmtSelect::Private
{
public:
   // blocks are kept separate for easier destruction and accounting.
   typedef std::deque<SynTree*> BlockList;   

   // blocks ordered for int
   typedef std::map<int64, SynTree*> IntBlocks;

   // blocks ordered for class -- used only in case of perfect matches.
   typedef std::map<Class*, SynTree*> ClassBlocks;

   // Declaration-ordered list of classes.
   typedef std::vector< Class* > ClassList;

   IntBlocks m_intBlocks;
   ClassBlocks m_classBlocks;
   
   BlockList m_blocks;
   ClassList m_classList;


   Private() {}
   
   Private( StmtSelect* owner, const Private& other ) 
   {
      IntBlocks::const_iterator ibi = other.m_intBlocks.begin();
      while( ibi != other.m_intBlocks.end() )
      {
         SynTree* st = ibi->second->clone();
         st->setParent(owner);
         m_intBlocks[ibi->first] = st;
         m_blocks.push_back( st );
         ++ibi;
      }
      
      ClassBlocks::const_iterator cbi = other.m_classBlocks.begin();
      while( cbi != other.m_classBlocks.end() )
      {
         SynTree* st = cbi->second->clone();
         st->setParent(owner);
         m_classBlocks[cbi->first] = st;
         m_blocks.push_back( st );
         ++cbi;
      }
    
      // class list can be a flat copy, classes are in the engine.
      m_classList = other.m_classList;
   }
   
   ~Private()
   {
      BlockList::iterator iter = m_blocks.begin();
      while( iter != m_blocks.end() )
      {
         delete *iter;
         ++iter;
      }      
   }
};


StmtSelect::StmtSelect( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private ),
   m_expr( expr ),
   m_defaultBlock(0),
   m_unresolved(0)
{
   FALCON_DECLARE_SYN_CLASS( stmt_select );

   if (expr != 0 )
   {
      expr->setParent(this);
      apply = apply_;
   }
   else
   {
      // we're used just as a dictionary.
      apply = 0;
   }
}

StmtSelect::StmtSelect( const StmtSelect& other ):
   Statement( other ),
   m_expr(0),
   m_defaultBlock(0),
   m_unresolved( other.m_unresolved )
{
   // we can't duplicate unresolved symbols.
   fassert( m_unresolved == 0 );
   
   if ( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
      apply = apply_;
   }
   else
   {
      // we're used just as a dictionary.
      apply = 0;
   }

   if ( other.m_defaultBlock != 0 )
   {
      m_defaultBlock = other.m_defaultBlock->clone();
      m_defaultBlock->setParent(this);
   }
   
   _p = new Private( this, *other._p );
}

StmtSelect::~StmtSelect()
{
   delete m_expr;
   delete m_defaultBlock;
   delete _p;
}


void StmtSelect::describeTo( String& tgt, int depth ) const
{
   if( m_expr != 0 )
   {
      String prefix = String(" ").replicate( depth * depthIndent );
      tgt = prefix + "select " + m_expr->describe() +"\n";
   }

   //TODO...
}


Expression* StmtSelect::selector() const
{
   return m_expr;
}


bool StmtSelect::selector( Expression* expr )
{
   if( expr == 0 )
   {
      delete m_expr;
      m_expr = 0;
      // we're used just as a dictionary.
      apply = 0;
   }
   else {
      if( ! expr->setParent(this) ) return false;
      delete m_expr;
      m_expr = expr;
      apply = apply_;
   }
   return true;
}

void StmtSelect::oneLinerTo( String& tgt ) const
{
   if( m_expr != 0 )
   {
      tgt = "select " + m_expr->oneLiner();
   }
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
      if( ! block->setParent(this) ) return false;
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
      if( ! block->setParent(this) ) return false;
      _p->m_blocks.push_back( block );
   }

   // anyhow, associate the request.
   _p->m_classBlocks[ cls ] = block;

   // Add the class in order -- as already resolved.   
   _p->m_classList.push_back( cls );

   return true;
}

Requirement* StmtSelect::addSelectName( const String& name, SynTree* block )
{
   // save the block only if not just pushed in the last operation.
   if( _p->m_blocks.empty() || _p->m_blocks.back() != block )
   {
      if( ! block->setParent(this) ) return 0;
      _p->m_blocks.push_back( block );
   }

   // add an empty class as a placeholder.
   _p->m_classList.push_back( 0 );
   m_unresolved++;
   
   // add a requirement pointing to the missing class.
   SelectRequirement* req = new SelectRequirement( 
         _p->m_blocks.size()-1, _p->m_classList.size()-1, block->sr().line(), name, this );
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
      Class* base = (*iter);
      // base can be 0 if the class was resolved as actually being an integer.
      if( base != 0 && cls->isDerivedFrom( base ) )
      {
         Private::ClassBlocks::iterator pos = _p->m_classBlocks.find( cls );
         fassert( pos != _p->m_classBlocks.end() );
         return pos->second;
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
   if( ! block->setParent(this) )
   {
      return false;
   }

   delete m_defaultBlock;
   m_defaultBlock = block;
   return true;
}

bool StmtSelect::setSelectClass( int id, int clsId, Class* cls )
{
   fassert( clsId < (int) _p->m_classList.size() );
   fassert( id < (int) _p->m_blocks.size() );
   
   // refuse to add if already existing.
   if ( _p->m_classBlocks.find( cls ) != _p->m_classBlocks.end() )
   {
      return false;
   }

   // anyhow, associate the request.
   _p->m_classBlocks[ cls ] = _p->m_blocks[id];
   // and fix the ID in the class list.
   _p->m_classList[clsId] = cls;
   
   m_unresolved--;
   return true;
}

bool StmtSelect::setSelectType( int id, int typeId )
{
   fassert( id < (int) _p->m_blocks.size() );
   
   // refuse to add if already existing.
   if ( _p->m_intBlocks.find( typeId ) != _p->m_intBlocks.end() )
   {
      return false;
   }

   // anyhow, associate the request.
   _p->m_intBlocks[ typeId ] = _p->m_blocks[id];
   // -- there's nothing to fix in the class list, leave it 0.
   m_unresolved--;
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

void SelectRequirement::onResolved(
         const Module* source, const Symbol* sym, Module* requirer, Symbol*  )
{
   fassert( m_owner == 0 );
   
   const Item* itm = sym->getValue(0);
   if( itm == 0 || (!itm->isOrdinal()&& ! itm->isClass()) )
   {
      throw new LinkError( ErrorParam( m_owner->selector() == 0 ? e_catch_invtype : e_select_invtype )
         .line( m_line )
         .module( requirer->uri() )
         .origin( ErrorParam::e_orig_linker )
         .symbol( sym->name() )
         .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
         );
   }

   // an integer?
   if( itm->isOrdinal() )
   {
      int64 tid = itm->forceInteger();
      if( ! m_owner->setSelectType( m_id, tid ) )
      {
         throw new LinkError( ErrorParam( m_owner->selector() == 0 ? e_catch_clash : e_switch_clash )
            .line( m_line )
            .module( requirer->uri() )
            .origin( ErrorParam::e_orig_linker )
            .symbol( sym->name() )
            .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
            );
      }      
   }
   else
   {
      fassert( itm->asClass()->isMetaClass() );
      Class* cls = static_cast<Class*>(itm->asInst());
      if( ! m_owner->setSelectClass( m_id, m_clsId, cls ) )
      {
         throw new LinkError( ErrorParam( m_owner->selector() == 0 ? e_catch_clash : e_switch_clash )
            .line( m_line )
            .module( requirer->uri() )
            .origin( ErrorParam::e_orig_linker )
            .symbol( sym->name() )
            .extra( String("declared in ") + (source != 0 ? source->uri() : "<internal>" ) )
            );
      }
   }
}


class SelectRequirement::ClassSelectRequirement: public ClassRequirement
{
public:
   ClassSelectRequirement():
      ClassRequirement("$SelectRequirement")
   {}

   virtual ~ClassSelectRequirement() {}
   
   virtual void store( VMContext*, DataWriter* stream, void* instance ) const
   {
      SelectRequirement* s = static_cast<SelectRequirement*>(instance);
      s->store( stream );
   }
   
   virtual void flatten( VMContext*, ItemArray& subItems, void* instance ) const
   {
      SelectRequirement* s = static_cast<SelectRequirement*>(instance);
      subItems.append( Item(s->m_owner->handler(), s->m_owner ) );
   }
   
   virtual void unflatten( VMContext*, ItemArray& subItems, void* instance ) const
   {
      SelectRequirement* s = static_cast<SelectRequirement*>(instance);
      fassert( subItems.length() == 1 );
      fassert( subItems[0].asClass()->name() == "Select" );
      s->m_owner = static_cast<StmtSelect*>(subItems[0].asInst());
   }
   
   virtual void restore( VMContext*, DataReader* stream, void*& empty ) const
   {
      SelectRequirement* s = 0;
      try {
         s = new SelectRequirement(0,0,0,"",0);
         s->restore(stream);
         empty = s;
      }
      catch( ... )
      {
         delete s;
         throw;
      }
   }
   
   void describe( void* instance, String& target, int, int ) const
   {
      SelectRequirement* s = static_cast<SelectRequirement*>(instance);
      if( s->m_owner == 0 )
      {
         target = "<Blank SelectRequirement>";
      }
      else {
         target = "SelectRequirement for \"" + s->name() + "\"";
      }
   }
};




Class* SelectRequirement::cls() const
{
   static Class* theClass = new ClassSelectRequirement;
   static bool bRegistered = false;

   // a double registration in MT context has no adverse effect.
   // we just need to have it in the engine to let it to delete it at end.
   if( ! bRegistered ) {
      bRegistered = true;
      Engine::instance()->addMantra(theClass);
   }

   return theClass;
}

}

/* end of stmtselect.cpp */

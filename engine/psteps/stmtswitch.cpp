/*
   FALCON - The Falcon Programming Language.
   FILE: stmtswitch.cpp

   Syntactic tree item definitions -- switch statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 May 2012 16:33:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtswitch.cpp"

#include <falcon/expression.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>

#include <falcon/errors/linkerror.h>

#include <falcon/psteps/stmtswitch.h>
#include <falcon/classes/classrequirement.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/itemarray.h>

#include <map>
#include <set>
#include <vector>
#include <algorithm>

namespace Falcon {

class StmtSwitch::Private
{
public:
   // blocks are kept separate for easier destruction and replication.
   typedef std::vector<SynTree*> BlockVector; 
   typedef std::map<SynTree*, size_t> BlockSet;   
   
   typedef std::pair <int64, size_t> BlockIntLimit; 
   typedef std::pair <String, size_t> BlockStringLimit; 

   // blocks ordered for int
   typedef std::map<int64, BlockIntLimit> IntBlocks;

   // blocks ordered for strings.
   typedef std::map<String, BlockStringLimit> StringBlocks;

   // Declaration-ordered list of discrete variable values.
   typedef std::pair<Symbol*, size_t> VarBlockLimit;
   typedef std::vector< VarBlockLimit > ItemBlockList;

   IntBlocks m_intBlocks;
   StringBlocks m_stringBlocks;   
   ItemBlockList m_itemBlocks;   
   BlockVector m_blocks;
   BlockSet m_blocksSet;

   Private() {}
   
   Private( StmtSwitch* owner, const StmtSwitch::Private& other ) 
   {
      // deep copy of the syntree references.
      const BlockVector& bl = other.m_blocks;
      {
         BlockVector::const_iterator pos = bl.begin();
         while( pos != bl.end() ) {
            SynTree* copy = (*pos)->clone();
            copy->setParent(owner);
            m_blocks.push_back( copy );
            ++pos;
         }
      }
      
      // flat copy of the rest.
      m_blocksSet = other.m_blocksSet;
      m_intBlocks = other.m_intBlocks;
      m_stringBlocks = other.m_stringBlocks;
      m_itemBlocks = other.m_itemBlocks;
   }
   
   ~Private()
   {
      BlockVector::iterator iter = m_blocks.begin();
      while( iter != m_blocks.end() )
      {
         delete *iter;
         ++iter;
      }
   }
   
   void gcMark( uint32 mark ) 
   {
      BlockVector::iterator iter = m_blocks.begin();
      while( iter != m_blocks.end() )
      {
         SynTree* st = *iter;
         st->gcMark( mark );
         ++iter;
      }
      
      ItemBlockList::iterator bli = m_itemBlocks.begin();
      while( bli != m_itemBlocks.end() ) {
         VarBlockLimit& blv = *bli;
         blv.first->gcMark( mark );
         ++bli;
      }      
   }
};


StmtSwitch::StmtSwitch( Expression* expr, int32 line, int32 chr ):
   SwitchlikeStatement( line, chr ),
   _p( new Private ),
   m_expr( expr ),
   m_nilBlock(0),
   m_trueBlock(0),
   m_falseBlock(0)
{
   FALCON_DECLARE_SYN_CLASS( stmt_switch );
   apply = apply_;
   
   if (expr != 0 )
   {
      expr->setParent(this);      
   }
}

StmtSwitch::StmtSwitch( const StmtSwitch& other ):
   SwitchlikeStatement( other ),
   m_expr(0),
   m_nilBlock(0),
   m_trueBlock(0),
   m_falseBlock(0)
{
   apply = apply_;
   
   if ( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
   }  

   if ( other.m_defaultBlock != 0 )
   {
      m_defaultBlock = other.m_defaultBlock->clone();
      m_defaultBlock->setParent(this);
   }
   
   if ( other.m_nilBlock != 0 )
   {
      m_nilBlock = other.m_nilBlock->clone();
      m_nilBlock->setParent(this);
   }
   
   if ( other.m_trueBlock != 0 )
   {
      m_trueBlock = other.m_trueBlock->clone();
      m_trueBlock->setParent(this);
   }
   
   if ( other.m_falseBlock != 0 )
   {
      m_falseBlock = other.m_falseBlock->clone();
      m_falseBlock->setParent(this);
   }
   
   _p = new Private( this, *other._p );
}


StmtSwitch::~StmtSwitch()
{
   delete m_expr;
   delete m_trueBlock;
   delete m_falseBlock;
   delete m_nilBlock;
   
   delete _p;
}


void StmtSwitch::describeTo( String& tgt, int depth ) const
{
   if( m_expr == 0 )
   {
      tgt = "<Blank switch>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt = prefix + "switch " + m_expr->describe() +"\n";
   
   
}


Expression* StmtSwitch::selector() const
{
   return m_expr;
}


bool StmtSwitch::selector( Expression* expr )
{
   if( expr == 0 )
   {
      delete m_expr;
      m_expr = 0;      
   }
   else {
      if( ! expr->setParent(this) ) return false;
      delete m_expr;
      m_expr = expr;     
   }
   return true;
}

void StmtSwitch::oneLinerTo( String& tgt ) const
{
   if( m_expr != 0 )
   {
      tgt = "switch " + m_expr->oneLiner();
   }
}


bool StmtSwitch::addNilBlock( SynTree* block )
{
   if( m_nilBlock != 0 ) 
   {
      return false;
   }
   m_nilBlock = block;
   // ignore the result, we might have already added this elsewhere.
   block->setParent(this);
   return true;
}


bool StmtSwitch::addBoolBlock( bool value, SynTree* block )
{
   if( value ) {
      if( m_trueBlock == 0 ) {
         return false;
      }
      m_trueBlock = block;
   }
   else {
      if( m_falseBlock == 0 ) {
         return false;
      }
      m_falseBlock = block;
   }
   // ignore the result, we might have already added this elsewhere.
   block->setParent(this);   
   return true;
}


bool StmtSwitch::addIntBlock( int64 iValue, SynTree* block )
{
   return addRangeBlock( iValue, iValue, block );   
}


bool StmtSwitch::addRangeBlock( int64 iLow, int64 iHigh, SynTree* block )
{   
   // see if we have this block.
   Private::IntBlocks::iterator ithigh = _p->m_intBlocks.lower_bound( iHigh );
   
   if(ithigh == _p->m_intBlocks.end() || ithigh->second.first > iLow ) 
   {
      // ok, proceed
      size_t blockID;
      Private::BlockSet::iterator blockPos = _p->m_blocksSet.find( block );
      if( blockPos == _p->m_blocksSet.end() ) {
         blockID = _p->m_blocks.size();
         // TODO: Check result?
         block->setParent( this );
         
         _p->m_blocks.push_back( block );
         _p->m_blocksSet[block] = blockID;         
      }
      else {
         blockID = blockPos->second;
      }
      _p->m_intBlocks[iHigh] = std::make_pair( iLow, blockID );
      
      return true;
   }
   
   // Nope, the data is taken.
   return false;
}


bool StmtSwitch::addStringBlock( const String& strLow, SynTree* block )
{
   return addStringRangeBlock( strLow, strLow, block );
}


bool StmtSwitch::addStringRangeBlock( const String& strLow, const String& strHigh, SynTree* block )
{
   // see if we have this block.
   Private::StringBlocks::iterator ithigh = _p->m_stringBlocks.lower_bound( strHigh );
   
   if( ithigh == _p->m_stringBlocks.end() || ithigh->second.first > strLow ) 
   {
      // ok, proceed
      size_t blockID;
      Private::BlockSet::iterator blockPos = _p->m_blocksSet.find( block );
      if( blockPos == _p->m_blocksSet.end() ) {
         blockID = _p->m_blocks.size();
         // TODO: Check result?
         block->setParent( this );
         
         _p->m_blocks.push_back( block );
         _p->m_blocksSet[block] = blockID;         
      }
      else {
         blockID = blockPos->second;
      }
      _p->m_stringBlocks[strLow] = std::make_pair( strHigh, blockID );
      
      return true;
   }
   
   // Nope, the data is taken.
   return false;
}


bool StmtSwitch::addSymbolBlock( Symbol* var, SynTree* block )
{
   // check if the block is already there.
   Private::ItemBlockList::iterator bli = _p->m_itemBlocks.begin();
   while( bli != _p->m_itemBlocks.end() ) {
      Private::VarBlockLimit& blk = *bli;
      if( var == blk.first ) {
         return false;
      }
      bli++;
   }
   
   // ok, proceed
   size_t blockID;
   Private::BlockSet::iterator blockPos = _p->m_blocksSet.find( block );
   if( blockPos == _p->m_blocksSet.end() ) {
      blockID = _p->m_blocks.size();
      // TODO: Check result?
      block->setParent( this );

      _p->m_blocks.push_back( block );
      _p->m_blocksSet[block] = blockID;         
   }
   else {
      blockID = blockPos->second;
   }
   
   _p->m_itemBlocks.push_back( std::make_pair( var, blockID ) );
   return true;
}


SynTree* StmtSwitch::findBlockForNumber( int64 value ) const
{
   // see if we have this block.
   Private::IntBlocks::iterator it = _p->m_intBlocks.lower_bound( value );
   
   if( it == _p->m_intBlocks.end() || it->second.first > value ) 
   {
      return m_defaultBlock;
   }
   return _p->m_blocks[ it->second.second ];
}


SynTree* StmtSwitch::findBlockForString( const String& value ) const
{
   // see if we have this block.
   Private::StringBlocks::iterator it = _p->m_stringBlocks.lower_bound( value );
   
   if( it == _p->m_stringBlocks.end() || it->second.first > value ) 
   {  
      return m_defaultBlock;
   }
   
   return _p->m_blocks[ it->second.second ];
}


SynTree* StmtSwitch::findBlockForItem( const Item& value ) const
{
   switch( value.type() ) {
      case FLC_ITEM_NIL:
         if( m_nilBlock ) {
            return m_nilBlock;
         }
         break;
         
      case FLC_ITEM_BOOL:
         if( value.asBoolean() ) {
            if( m_trueBlock ) { return m_trueBlock; }
         }
         else {
            if( m_falseBlock ) { return m_trueBlock; }
         }
         break;
         
      case FLC_ITEM_INT: case FLC_ITEM_NUM:
      {
         int64 vl = value.forceInteger();
         // see if we have this block.
         Private::IntBlocks::iterator it = _p->m_intBlocks.lower_bound( vl );

         if( it != _p->m_intBlocks.end() ) 
         {
            register int64 low = it->second.first;
            if( low <= vl ) {
               return _p->m_blocks[ it->second.second ];
            }
         }
      }
      break;
      
      case FLC_ITEM_USER:
      if( value.isString() ) {
         String* vl = value.asString();
         // see if we have this block.
         Private::StringBlocks::iterator it = _p->m_stringBlocks.lower_bound( *vl );

         if( it != _p->m_stringBlocks.end() && it->second.first >= *vl ) 
         {
            return _p->m_blocks[ it->second.second ];
         }         
      }
      break;
   }   
   
   return 0;
}


void StmtSwitch::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtSwitch* self = static_cast<const StmtSwitch*>(ps);

   CodeFrame& cf = ctx->currentCode();
   // first time around? -- call the expression.
   switch( cf.m_seqId )
   {
      case 0:
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_expr, cf ) )
         {
            return;
         }
         //fallthrough
        
      case 1:         
         {
            SynTree* selectedBlock = self->findBlockForItem( ctx->topData() );
            // found?
            if( selectedBlock != 0 ) {
               // -- we're done.
               ctx->popCode();
               ctx->popData();
               ctx->stepIn( selectedBlock );
               return;
            }
         }
         cf.m_seqId = 2;
         // falltrough
         
      default:
      {            
         Private::ItemBlockList& lst = self->_p->m_itemBlocks;
         // eventually descend if we need to descend to do a compare.
         if( cf.m_seqId > 2 ) {
            Private::VarBlockLimit& lmt = lst[cf.m_seqId-3]; // already advanced by 1
            bool isEqual = ctx->topData().asInteger() == 0;
            if( isEqual ) {
               // found!
               ctx->popCode();
               ctx->popData(2);
               ctx->stepIn(self->_p->m_blocks[lmt.second]);
               return;
            }
         }
         
         Item copy = ctx->topData();
         Class* cls; 
         void* data;
         copy.forceClassInst( cls, data );
         
         while( ((unsigned)cf.m_seqId) < lst.size() + 2 ) {
            Private::VarBlockLimit& lmt = lst[cf.m_seqId-2];
            cf.m_seqId++;
            Symbol* sym = lmt.first;
            const Item* vcomp = sym->getValue(ctx);
            // found?
            if( vcomp != 0 )
            {               
               ctx->pushData( copy );
               ctx->pushData( *vcomp );
               // launch te compare.
               cls->op_compare( ctx, data );
               // Went deep?
               if( &cf != &ctx->currentCode() ) {
                  return;
               }
               
               bool isEqual = ctx->topData().asInteger() == 0;
               if( isEqual ) {
                  // found!
                  ctx->popCode();
                  ctx->popData(2);
                  ctx->stepIn(self->_p->m_blocks[lmt.second]);
                  return;
               }
               // try next
               ctx->popData();
            }
         }                  
      }
   }

   // nope, we didn't find it.
   // anyway, we're done...
   ctx->popCode();
   ctx->popData();
   //... eventually push in the default block
   if( self->m_defaultBlock ) {
      ctx->stepIn(self->m_defaultBlock);
   }
}


void StmtSwitch::gcMark( uint32 mark )
{
   if( m_gcMark != mark ) {
      m_gcMark = mark;
      _p->gcMark( mark );
   }
}

}

/* end of stmtswitch.cpp */

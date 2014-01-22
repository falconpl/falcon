/*
   FALCON - The Falcon Programming Language.
   FILE: importdef.cpp

   Structure recording the import definition in modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Nov 2011 14:16:51 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/importdef.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <vector>

namespace Falcon {

class ImportDef::SymbolList: public std::vector<String>
{};


ImportDef::ImportDef():
   m_symList( 0 ),
   m_bIsLoad( false ),
   m_bIsUri( false ),
   m_bIsNS( false ),
   m_isProcessed(false),
   m_isLoaded(false),
   m_modreq(0),
   m_id(0)
{}


ImportDef::ImportDef( const String& path, bool isFsPath, const String& symName,
                     const String& nsName, bool bIsNS ):
   m_symList( 0 ),
   m_bIsLoad( false ),
   m_bIsUri( false ),
   m_bIsNS( false ),
   m_isProcessed(false),
   m_isLoaded(false),
   m_modreq(0),
   m_id(0)
{
   setImportFrom( path, isFsPath, symName, nsName, bIsNS );      
}
   

ImportDef::~ImportDef()
{
   delete m_symList;
}


void ImportDef::setTargetNS( const String& ns )
{
   m_bIsNS = true;
   m_tgNameSpace = ns;
}


void ImportDef::setTargetSymbol( const String &sym )
{
   m_bIsNS = false;
   m_tgNameSpace = sym;
}


ImportDef& ImportDef::addSourceSymbol( const String& sym )
{
   if( m_symList == 0 )
   {
      m_symList = new SymbolList;
   }
   
   m_symList->push_back( sym );
   return *this;
}


void ImportDef::setImportModule( const String& src, bool bIsPath )
{
   m_bIsLoad = false;
   m_bIsUri = bIsPath;
   m_module = src;
}


void ImportDef::setLoad( const String& src, bool bIsPath )
{
   m_bIsLoad = true;
   m_bIsUri = bIsPath;
   m_module = src;
}


bool ImportDef::setImportFrom( const String& path, bool isFsPath, const String& symName,
   const String& nsName, bool bIsNS )
{
   
   m_bIsLoad = false;
   m_bIsUri = isFsPath;
   m_bIsNS = bIsNS;
   
   m_module = path;
   if( m_symList == 0 )
   {
      if( symName.size() != 0 )
      {
         m_symList = new SymbolList;
      }
   }
   else
   {
      m_symList->clear();
   }
   
   if( symName.size() != 0 )
   {
      m_symList->push_back( symName );
   }
   
   m_tgNameSpace = nsName;
   return true;
}

   
int ImportDef::symbolCount() const
{
   if( m_symList == 0 )
   {
      return 0;
   }
   
   return (int)m_symList->size();
}


const String& ImportDef::sourceSymbol( int n ) const
{
   static String ss;
   if( m_symList == 0 || n < 0 || n >= (int)m_symList->size() )
   {
      return ss;
   }
   
   return m_symList->at( n );
}


void ImportDef::targetSymbol( int i, String& target ) const
{
   if ( m_symList == 0 || m_symList->size() <= (unsigned) i )
   {
      target.size(0);
      return;
   }
   
   if( m_bIsNS )
   {
      // we must add the namespace...
      target = m_tgNameSpace;

      // possibly removing the old one... 
      const String& name = m_symList->at(i);
      length_t pos = name.find('.');
      if( pos == String::npos )
      {
         target += "." + name;
      }
      else
      {
         target += name.subString(pos);
      }      
   }
   else
   {
      // is it an "as symbol" ?
      if( m_tgNameSpace.size() > 0 )
      {
         target = m_tgNameSpace;
      }
      else
      {
         // the import respects source namespaces.
         target = m_symList->at(i);
      }
   }
}


bool ImportDef::isValid() const
{
   if( m_bIsLoad ) {
      return m_module.size() != 0;
   }
   else
   {
      return m_bIsNS || m_tgNameSpace.size() == 0 || m_symList->size() == 1;
   }
}
   

void ImportDef::describe( String& tgt ) const
{
   if( ! isValid() )
   {
      tgt = "invalid import clause";
   }
   
   if( m_bIsLoad ) {
      tgt = "load ";
      if( m_bIsUri )
      {
         tgt += '"';
         tgt += m_module;
         tgt += '"';
      }
      else
      {
         tgt += m_module;
      }
   }
   else
   {
      tgt = "import ";
      SymbolList::const_iterator iter = m_symList->begin();
      
      while( iter != m_symList->end() )
      {
         tgt += *iter;
         ++iter;
         if ( iter != m_symList->end() )
         {
            tgt += ", ";
         }
      }
      
      if( m_module.size() != 0 ) {
         tgt += " from " + m_module;
      }
      
      if( m_tgNameSpace.size() != 0 )
      {
         tgt += (m_bIsNS ? " in " : " as " );
         tgt += m_tgNameSpace;
      }
   }
}


void ImportDef::store(DataWriter* wr) const
{
   wr->write( m_bIsLoad );
   wr->write( m_bIsUri );
   wr->write( m_bIsNS );

   wr->write( m_module );
   wr->write( m_tgNameSpace );
   m_sr.serialize(wr);

   // write the symbols
   uint32 symSize = m_symList == 0 ? 0 : m_symList->size();
   wr->write( symSize );

   for ( size_t i = 0; i < symSize; ++i )
   {
      const String& name = m_symList->at(i);
      wr->write(name);
   }
}


void ImportDef::restore( DataReader* rd ) 
{
   rd->read( m_bIsLoad );
   rd->read( m_bIsUri );
   rd->read( m_bIsNS );

   rd->read( m_module );
   rd->read( m_tgNameSpace );
   
   m_sr.deserialize(rd);

   uint32 symSize;
   rd->read( symSize );
   if( symSize != 0 )
   {
      m_symList = new SymbolList;
      for ( size_t i = 0; i < symSize; ++i )
      {
         String symName;
         rd->read(symName);
         m_symList->push_back( symName );
      }
   }
}

   
}

/* end of importdef.cpp */

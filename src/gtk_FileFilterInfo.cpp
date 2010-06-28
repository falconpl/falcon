/**
 *  \file gtk_FileFilterInfo.cpp
 */

#include "gtk_FileFilterInfo.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void FileFilterInfo::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_FileFilterInfo = mod->addClass( "%GtkFileFilterInfo" );

    c_FileFilterInfo->setWKS( true );
    c_FileFilterInfo->getClassDef()->factory( &FileFilterInfo::factory );

    mod->addClassProperty( c_FileFilterInfo, "contains" );
    mod->addClassProperty( c_FileFilterInfo, "filename" );
    mod->addClassProperty( c_FileFilterInfo, "uri" );
    mod->addClassProperty( c_FileFilterInfo, "display_name" );
    mod->addClassProperty( c_FileFilterInfo, "mime_type" );
}


FileFilterInfo::FileFilterInfo( const Falcon::CoreClass* gen, const GtkFileFilterInfo* info )
    :
    Falcon::CoreObject( gen ),
    m_info( NULL )
{
    if ( info )
        m_info = (GtkFileFilterInfo*) info;
}


FileFilterInfo::~FileFilterInfo()
{
}


bool FileFilterInfo::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "contains" )
        it = (int64) m_info->contains;
    else
    if ( s == "filename" )
        it = UTF8String( m_info->filename ? m_info->filename : "" );
    else
    if ( s == "uri" )
        it = UTF8String( m_info->uri ? m_info->uri : "" );
    else
    if ( s == "display_name" )
        it = UTF8String( m_info->display_name ? m_info->display_name : "" );
    else
    if ( s == "mime_type" )
        it = UTF8String( m_info->mime_type ? m_info->mime_type : "" );
    else
        return false;
    return true;
}


bool FileFilterInfo::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* FileFilterInfo::factory( const Falcon::CoreClass* gen, void* info, bool )
{
    return new FileFilterInfo( gen, (GtkFileFilterInfo*) info );
}


/*#
    @class GtkFileFilterInfo
    @brief A GtkFileFilterInfo struct is used to pass information about the tested file to gtk_file_filter_filter().

    @prop contains Flags indicating which of the following fields need are filled (GtkFileFilterFlags)
    @prop filename the filename of the file being tested
    @prop uri the URI for the file being tested
    @prop display_name the string that will be used to display the file in the file chooser
    @prop mime_type the mime type of the file
 */


} // Gtk
} // Falcon

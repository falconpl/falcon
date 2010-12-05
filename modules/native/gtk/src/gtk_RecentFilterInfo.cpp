/**
 *  \file gtk_RecentFilterInfo.cpp
 */

#include "gtk_RecentFilterInfo.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RecentFilterInfo::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RecentFilterInfo = mod->addClass( "%GtkRecentFilterInfo" );

    c_RecentFilterInfo->setWKS( true );
    c_RecentFilterInfo->getClassDef()->factory( &RecentFilterInfo::factory );

    mod->addClassProperty( c_RecentFilterInfo, "contains" );
    mod->addClassProperty( c_RecentFilterInfo, "uri" );
    mod->addClassProperty( c_RecentFilterInfo, "display_name" );
    mod->addClassProperty( c_RecentFilterInfo, "mime_type" );
    mod->addClassProperty( c_RecentFilterInfo, "applications" );
    mod->addClassProperty( c_RecentFilterInfo, "groups" );
    mod->addClassProperty( c_RecentFilterInfo, "age" );
}


RecentFilterInfo::RecentFilterInfo( const Falcon::CoreClass* gen, const GtkRecentFilterInfo* info )
    :
    Falcon::CoreObject( gen ),
    m_info( NULL )
{
    if ( info )
        m_info = (GtkRecentFilterInfo*) info;
}


RecentFilterInfo::~RecentFilterInfo()
{
}


bool RecentFilterInfo::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "contains" )
        it = (int64) m_info->contains;
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
    if ( s == "applications" )
    {
        if ( m_info->applications )
        {
            int cnt = 0;
            gchar* p;
            for ( p = (gchar*) m_info->applications[0]; p; ++p ) ++cnt;
            CoreArray* arr = new CoreArray( cnt );
            for ( p = (gchar*) m_info->applications[0]; p; )
                arr->append( UTF8String( p++ ) );
            it = arr;
        }
        else
            it = new CoreArray( 0 );
    }
    else
    if ( s == "groups" )
    {
        if ( m_info->groups )
        {
            int cnt = 0;
            gchar* p;
            for ( p = (gchar*) m_info->groups[0]; p; ++p ) ++cnt;
            CoreArray* arr = new CoreArray( cnt );
            for ( p = (gchar*) m_info->groups[0]; p; )
                arr->append( UTF8String( p++ ) );
            it = arr;
        }
        else
            it = new CoreArray( 0 );
    }
    else
    if ( s == "age" )
        it = m_info->age;
    else
        return false;
    return true;
}


bool RecentFilterInfo::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* RecentFilterInfo::factory( const Falcon::CoreClass* gen, void* info, bool )
{
    return new RecentFilterInfo( gen, (GtkRecentFilterInfo*) info );
}


/*#
    @class GtkRecentFilterInfo
    @brief A GtkRecentFilterInfo struct is used to pass information about the tested file to gtk_recent_filter_filter().

    @prop contains Flags indicating which of the following fields need are filled (GtkRecentFilterFlags)
    @prop uri the URI for the file being tested
    @prop display_name the string that will be used to display the file in the recent chooser
    @prop mime_type the mime type of the file
    @prop applications  the list of applications that have registered the file
    @prop groups the groups to which the file belongs to
    @prop age the number of days elapsed since the file has been registered
 */


} // Gtk
} // Falcon

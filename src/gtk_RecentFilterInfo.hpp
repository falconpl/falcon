#ifndef GTK_RECENTFILTERINFO_HPP
#define GTK_RECENTFILTERINFO_HPP

#include "modgtk.hpp"

#define GET_RECENTFILTERINFO( item ) \
        (((Gtk::RecentFilterInfo*) (item).asObjectSafe() )->getInfo())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RecentFilterInfo
 */
class RecentFilterInfo
    :
    public Falcon::CoreObject
{
public:

    RecentFilterInfo( const Falcon::CoreClass*, const GtkRecentFilterInfo* = 0 );

    ~RecentFilterInfo();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkRecentFilterInfo* getInfo() const { return (GtkRecentFilterInfo*) m_info; }

private:

    GtkRecentFilterInfo*    m_info;

};


} // Gtk
} // Falcon

#endif // !GTK_RECENTFILTERINFO_HPP

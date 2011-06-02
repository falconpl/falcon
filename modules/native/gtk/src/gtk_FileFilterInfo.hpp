#ifndef GTK_FILEFILTERINFO_HPP
#define GTK_FILEFILTERINFO_HPP

#include "modgtk.hpp"

#define GET_FILEFILTERINFO( item ) \
        (((Gtk::FileFilterInfo*) (item).asObjectSafe() )->getInfo())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::FileFilterInfo
 */
class FileFilterInfo
    :
    public Falcon::CoreObject
{
public:

    FileFilterInfo( const Falcon::CoreClass*, const GtkFileFilterInfo* = 0 );

    ~FileFilterInfo();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkFileFilterInfo* getInfo() const { return (GtkFileFilterInfo*) m_info; }

private:

    GtkFileFilterInfo*  m_info;

};


} // Gtk
} // Falcon

#endif // !GTK_FILEFILTERINFO_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;

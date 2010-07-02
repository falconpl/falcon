#ifndef GTK_FILEFILTER_HPP
#define GTK_FILEFILTER_HPP

#include "modgtk.hpp"

#define GET_FILEFILTER( item ) \
        ((GtkFileFilter*)((Gtk::FileFilter*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::FileFilter
 */
class FileFilter
    :
    public Gtk::CoreGObject
{
public:

    FileFilter( const Falcon::CoreClass*, const GtkFileFilter* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_name( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC add_mime_type( VMARG );

    static FALCON_FUNC add_pattern( VMARG );

    static FALCON_FUNC add_pixbuf_formats( VMARG );

    static FALCON_FUNC add_custom( VMARG );

    static gboolean exec_custom( const GtkFileFilterInfo*, gpointer );

    static FALCON_FUNC get_needed( VMARG );

    static FALCON_FUNC filter( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FILEFILTER_HPP

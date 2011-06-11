#ifndef GTK_FILECHOOSERBUTTON_HPP
#define GTK_FILECHOOSERBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::FileChooserButton
 */
class FileChooserButton
    :
    public Gtk::CoreGObject
{
public:

    FileChooserButton( const Falcon::CoreClass*, const GtkFileChooserButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if GTK_CHECK_VERSION( 2, 12, 0 )
    static FALCON_FUNC signal_file_set( VMARG );

    static void on_file_set( GtkFileChooserButton*, gpointer );
#endif

    //static FALCON_FUNC new_with_backend( VMARG );

    static FALCON_FUNC new_with_dialog( VMARG );

    static FALCON_FUNC get_title( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_width_chars( VMARG );

    static FALCON_FUNC set_width_chars( VMARG );

    static FALCON_FUNC get_focus_on_click( VMARG );

    static FALCON_FUNC set_focus_on_click( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FILECHOOSERBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;

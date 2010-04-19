#ifndef GTK_FILECHOOSERDIALOG_HPP
#define GTK_FILECHOOSERDIALOG_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::FileChooserDialog
 */
class FileChooserDialog
    :
    public Gtk::CoreGObject
{
public:

    FileChooserDialog( const Falcon::CoreClass*, const GtkFileChooserDialog* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FILECHOOSERDIALOG_HPP

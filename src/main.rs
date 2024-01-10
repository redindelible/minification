use std::fmt::Debug;
use std::{io, thread};
use std::io::{BufReader, Read, Write};
use std::ops::{Index, IndexMut};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use clap::{Parser, Subcommand};
use bitvec::prelude::*;
use rayon::prelude::*;
use image::GenericImageView;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Offset {
    x: isize,
    y: isize
}

impl Offset {
    // fn surrounding() -> [Offset; 8] {
    //     [
    //         Offset { x: 1, y: 0},
    //         Offset { x: 1, y: -1},
    //         Offset { x: 0, y: -1},
    //         Offset { x: -1, y: -1},
    //         Offset { x: -1, y: 0},
    //         Offset { x: -1, y: 1},
    //         Offset { x: 0, y: 1},
    //         Offset { x: 1, y: 1},
    //     ]
    // }

    fn cross() -> [Offset; 4] {
        [
            Offset { x: 1, y: 0},
            Offset { x: 0, y: -1},
            Offset { x: -1, y: 0},
            Offset { x: 0, y: 1},
        ]
    }

    fn down() -> Offset {
        Offset { x: 0, y: 1 }
    }

    fn up() -> Offset {
        Offset { x: 0, y: -1 }
    }

    fn right() -> Offset {
        Offset { x: 1, y: 0 }
    }

    fn left() -> Offset {
        Offset { x: -1, y: 0 }
    }

    fn left_down() -> Offset {
        Offset { x: -1, y: 1 }
    }

    fn left_up() -> Offset {
        Offset { x: -1, y: -1 }
    }

    fn right_down() -> Offset {
        Offset { x: 1, y: 1 }
    }

    fn right_up() -> Offset {
        Offset { x: 1, y: -1 }
    }

    fn reverse(self) -> Offset {
        Offset { x: -self.x, y: -self.y }
    }

    fn with(self, index: (usize, usize)) -> Option<(usize, usize)> {
        Some((index.0.checked_add_signed(self.x)?, index.1.checked_add_signed(self.y)?))
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
struct Buffer2D<T> {
    width: usize,
    height: usize,
    items: Vec<T>
}

impl<T> Buffer2D<T> {
    fn from_fill(width: usize, height: usize, item: T) -> Buffer2D<T> where T: Clone {
        Buffer2D { width, height, items: vec![item; width * height] }
    }

    fn from_fn(width: usize, height: usize, mut f: impl FnMut(usize, usize) -> T) -> Buffer2D<T> {
        let mut items = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                items.push(f(x, y));
            }
        }
        Buffer2D { width, height, items }
    }

    #[cfg(test)]
    fn from_array<const WIDTH: usize, const HEIGHT: usize>(arr: [[T; WIDTH]; HEIGHT]) -> Buffer2D<T> {
        let items = arr.into_iter().flatten().collect();
        Buffer2D { width: WIDTH, height: HEIGHT, items }
    }

    fn clear(&mut self, item: &T) where T: Clone {
        for slot in self.items.iter_mut() {
            slot.clone_from(item);
        }
    }

    fn get(&self, index: (usize, usize)) -> Option<&T> {
        if index.0 >= self.width {
            None
        } else {
            self.items.get(index.1 * self.width + index.0)
        }
    }

    fn get_offset(&self, index: (usize, usize), offset: Offset) -> Option<&T> {
        self.get(offset.with(index)?)
    }
}

impl<T> Index<(usize, usize)> for Buffer2D<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        &self.items[index.1 * self.width + index.0]
    }
}

impl<T> IndexMut<(usize, usize)> for Buffer2D<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.items[index.1 * self.width + index.0]
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
enum Edge {
    Right,
    RightDown,
    Down,
    LeftDown,
    Left,
    LeftUp,
    Up,
    RightUp,
}

impl Edge {
    fn new(offset: Offset) -> Edge {
        if offset == Offset::down() {
            Edge::Down
        } else if offset == Offset::up() {
            Edge::Up
        } else if offset == Offset::right() {
            Edge::Right
        } else if offset == Offset::left() {
            Edge::Left
        } else if offset == Offset::left_down() {
            Edge::LeftDown
        } else if offset == Offset::left_up() {
            Edge::LeftUp
        } else if offset == Offset::right_down() {
            Edge::RightDown
        } else if offset == Offset::right_up() {
            Edge::RightUp
        } else {
            panic!()
        }
    }

    fn offset(self) -> Offset {
        match self {
            Edge::Down => Offset::down(),
            Edge::Up => Offset::up(),
            Edge::Right => Offset::right(),
            Edge::Left => Offset::left(),
            Edge::LeftDown => Offset::left_down(),
            Edge::LeftUp => Offset::left_up(),
            Edge::RightDown => Offset::right_down(),
            Edge::RightUp => Offset::right_up(),
        }
    }

    fn to_bits(self) -> u8 {
        match self {
            Edge::Right => 0,
            Edge::RightDown => 1,
            Edge::Down => 2,
            Edge::LeftDown => 3,
            Edge::Left => 4,
            Edge::LeftUp => 5,
            Edge::Up => 6,
            Edge::RightUp => 7,
        }
    }

    fn from_bits(bits: u8) -> Edge {
        match bits {
            0 => Edge::Right,
            1 => Edge::RightDown,
            2 => Edge::Down,
            3 => Edge::LeftDown,
            4 => Edge::Left,
            5 => Edge::LeftUp,
            6 => Edge::Up,
            7 => Edge::RightUp,
            _ => panic!("{}", bits)
        }
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
enum EdgeDirection {
    Incoming,
    Outgoing,
}

#[derive(Debug)]
struct Edges {
    edges: Vec<(Edge, EdgeDirection)>
}

impl Edges {
    fn new() -> Edges {
        Edges {
            edges: Vec::new()
        }
    }

    fn add_outgoing(&mut self, offset: Offset) {
        self.edges.push((Edge::new(offset), EdgeDirection::Outgoing));
        self.edges.sort();
    }

    fn add_incoming(&mut self, offset: Offset) {
        self.edges.push((Edge::new(offset), EdgeDirection::Incoming));
        self.edges.sort();
    }

    fn pair_of_incoming(&self, incoming: Offset) -> Offset {
        if let Ok(index) = self.edges.binary_search(&(Edge::new(incoming), EdgeDirection::Incoming)) {
            if index == 0 {
                self.edges[self.edges.len()-1].0.offset()
            } else {
                self.edges[index - 1].0.offset()
            }
        } else {
            panic!()
        }
    }

    fn pair_of_outgoing(&self, outgoing: Offset) -> Offset {
        if let Ok(index) = self.edges.binary_search(&(Edge::new(outgoing), EdgeDirection::Outgoing)) {
            if index == self.edges.len()-1 {
                self.edges[0].0.offset()
            } else {
                self.edges[index + 1].0.offset()
            }
        } else {
            panic!()
        }
    }

    fn first_outgoing(&self) -> Option<Offset> {
        self.edges.iter().find(|(_, dir)| dir == &EdgeDirection::Outgoing).map(|(edge, _)| edge.offset())
    }
}

fn edges(component: &Buffer2D<bool>, index: (usize, usize)) -> Edges {
    let mut edges = Edges::new();
    if !component.get_offset(index, Offset::right()).unwrap_or(&false) {
        if *component.get_offset(index, Offset::right_up()).unwrap_or(&false) {
            edges.add_outgoing(Offset::right_up());
        } else if *component.get_offset(index, Offset::up()).unwrap_or(&false) {
            edges.add_outgoing(Offset::up());
        }

        if *component.get_offset(index, Offset::right_down()).unwrap_or(&false) {
            edges.add_incoming(Offset::right_down());
        } else if *component.get_offset(index, Offset::down()).unwrap_or(&false) {
            edges.add_incoming(Offset::down());
        }
    }

    if !component.get_offset(index, Offset::down()).unwrap_or(&false) {
        if *component.get_offset(index, Offset::right_down()).unwrap_or(&false) {
            edges.add_outgoing(Offset::right_down());
        } else if *component.get_offset(index, Offset::right()).unwrap_or(&false) {
            edges.add_outgoing(Offset::right());
        }

        if *component.get_offset(index, Offset::left_down()).unwrap_or(&false) {
            edges.add_incoming(Offset::left_down());
        } else if *component.get_offset(index, Offset::left()).unwrap_or(&false) {
            edges.add_incoming(Offset::left());
        }
    }

    if !component.get_offset(index, Offset::left()).unwrap_or(&false) {
        if *component.get_offset(index, Offset::left_down()).unwrap_or(&false) {
            edges.add_outgoing(Offset::left_down());
        } else if *component.get_offset(index, Offset::down()).unwrap_or(&false) {
            edges.add_outgoing(Offset::down());
        }

        if *component.get_offset(index, Offset::left_up()).unwrap_or(&false) {
            edges.add_incoming(Offset::left_up());
        } else if *component.get_offset(index, Offset::up()).unwrap_or(&false) {
            edges.add_incoming(Offset::up());
        }
    }

    if !component.get_offset(index, Offset::up()).unwrap_or(&false) {
        if *component.get_offset(index, Offset::left_up()).unwrap_or(&false) {
            edges.add_outgoing(Offset::left_up());
        } else if *component.get_offset(index, Offset::left()).unwrap_or(&false) {
            edges.add_outgoing(Offset::left());
        }

        if *component.get_offset(index, Offset::right_up()).unwrap_or(&false) {
            edges.add_incoming(Offset::right_up());
        } else if *component.get_offset(index, Offset::right()).unwrap_or(&false) {
            edges.add_incoming(Offset::right());
        }
    }

    edges
}

fn chain_code_component(component: &Buffer2D<bool>, top_left: (usize, usize)) -> Vec<Offset> {
    let top_left_edges = edges(component, top_left);

    let mut traversed = Vec::new();
    if let Some(start) = top_left_edges.first_outgoing() {
        let expected_incoming = top_left_edges.pair_of_outgoing(start);

        let mut curr_pos = start.with(top_left).unwrap();
        let mut direction = start;
        traversed.push(direction);
        while curr_pos != top_left || direction.reverse() != expected_incoming {
            let next_direction = edges(component, curr_pos).pair_of_incoming(direction.reverse());
            traversed.push(next_direction);
            curr_pos = next_direction.with(curr_pos).unwrap();
            direction = next_direction;
        }
    }

    traversed
}

trait WriteHelper {
    fn write_u8(&mut self, num: u8) -> io::Result<()>;
    fn write_u16(&mut self, num: u16) -> io::Result<()>;
    fn write_u24(&mut self, num: u32) -> io::Result<()>;
    fn write_u32(&mut self, num: u32) -> io::Result<()>;
}

impl<W: Write> WriteHelper for W {
    fn write_u8(&mut self, num: u8) -> io::Result<()> {
        self.write_all(&num.to_le_bytes())
    }

    fn write_u16(&mut self, num: u16) -> io::Result<()> {
        self.write_all(&num.to_le_bytes())
    }

    fn write_u24(&mut self, num: u32) -> io::Result<()> {
        self.write_all(&num.to_le_bytes()[0..3])
    }

    fn write_u32(&mut self, num: u32) -> io::Result<()> {
        self.write_all(&num.to_le_bytes())
    }
}

trait ReadHelper {
    fn read_u8(&mut self) -> io::Result<u8>;
    fn read_u16(&mut self) -> io::Result<u16>;
    fn read_u24(&mut self) -> io::Result<u32>;
    fn read_u32(&mut self) -> io::Result<u32>;

    fn read_n(&mut self, n: usize) -> io::Result<Vec<u8>>;
}

impl<R: Read> ReadHelper for R {
    fn read_u8(&mut self) -> io::Result<u8> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(u8::from_le_bytes(bytes))
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u24(&mut self) -> io::Result<u32> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes[0..3])?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_n(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0; n];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }
}

#[derive(Debug)]
struct Chain<C> {
    start: (usize, usize),
    color: C,
    chain: Vec<Offset>
}

impl Chain<bool> {
    fn to_bytes<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write(&(self.start.0 as u16).to_le_bytes()).unwrap();
        w.write(&(self.start.1 as u16).to_le_bytes()).unwrap();
        w.write(&[self.color as u8]).unwrap();

        w.write(&(self.chain.len() as u32).to_le_bytes()).unwrap();
        let mut chain_bits = bitvec::bitvec![u8, Lsb0;];
        for offset in &self.chain {
            chain_bits.extend_from_bitslice(&Edge::new(*offset).to_bits().view_bits::<Lsb0>()[0..3]);
        }
        w.write(&(chain_bits.as_raw_slice().len() as u32).to_le_bytes()).unwrap();
        w.write(chain_bits.as_raw_slice()).unwrap();

        Ok(())
    }

    fn from_bytes<R: Read>(r: &mut R) -> io::Result<Chain<bool>> {
        let start_x = r.read_u16().unwrap();
        let start_y = r.read_u16().unwrap();
        let color = r.read_u8().unwrap();

        let chain_length = r.read_u32().unwrap() as usize;
        let chain_bytes = r.read_u32().unwrap() as usize;
        let mut chain_bytes = vec![0; chain_bytes];
        r.read_exact(&mut chain_bytes).unwrap();
        let items: Vec<Offset> = BitVec::<_, Lsb0>::from_vec(chain_bytes)
            .chunks_exact(3)
            .map(|chunk| {
                Edge::from_bits(chunk.load_le::<u8>()).offset()
            })
            .take(chain_length)
            .collect();

        Ok(Chain {
            start: (start_x as usize, start_y as usize),
            color: color == 1,
            chain: items
        })
    }
}

#[derive(Debug)]
struct ChainEncoded<C> {
    size: (usize, usize),
    fill: C,
    chains: Vec<Chain<C>>
}

impl ChainEncoded<bool> {
    fn to_bytes<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write(&(self.size.0 as u16).to_le_bytes()).unwrap();
        w.write(&(self.size.1 as u16).to_le_bytes()).unwrap();
        w.write(&[self.fill as u8]).unwrap();

        w.write(&(self.chains.len() as u32).to_le_bytes()).unwrap();
        for chain in &self.chains {
            chain.to_bytes(w).unwrap();
        }

        Ok(())
    }

    fn from_bytes<R: Read>(r: &mut R) -> io::Result<ChainEncoded<bool>> {
        let width = r.read_u16().unwrap();
        let height = r.read_u16().unwrap();
        let fill = r.read_u8().unwrap();

        let chains_length = r.read_u32().unwrap();
        let mut chains = Vec::with_capacity(chains_length as usize);
        for _ in 0..chains_length {
            let chain = Chain::from_bytes(r).unwrap();
            chains.push(chain);
        }

        Ok(ChainEncoded {
            size: (width as usize, height as usize),
            fill: fill == 1,
            chains
        })
    }
}

fn chain_coding<C: Eq + Copy + Debug>(img: &Buffer2D<C>) -> ChainEncoded<C> {
    let mut chains: Vec<Chain<C>> = Vec::new();

    // extracted to not have to reallocate this every time
    let mut component = Buffer2D::from_fn(img.width, img.height, |_, _| false);
    let mut to_visit = Vec::with_capacity(1000000);

    let mut visited = Buffer2D::from_fn(img.width, img.height, |_, _| false);
    for y in 0..img.height {
        for x in 0..img.width {
            let index = (x, y);
            if !visited[index] {
                visited[index] = true;
                let component_color = img[index];
                component.clear(&false);

                to_visit.clear();
                to_visit.push(index);

                let mut top_leftmost = index;
                while let Some(index) = to_visit.pop() {
                    if index < top_leftmost {
                        top_leftmost = index;
                    }
                    component[index] = true;

                    for offset in Offset::cross() {
                        if let Some(other_color) = img.get_offset(index, offset) {
                            let offset_index = offset.with(index).unwrap();
                            if !visited[offset_index] && other_color.eq(&component_color) {
                                to_visit.push(offset_index);
                                visited[offset_index] = true;
                            }
                        };
                    }
                }

                let chain = chain_code_component(&component, top_leftmost);

                chains.push(Chain {
                    start: top_leftmost,
                    color: component_color,
                    chain
                });
            }
        }
    }

    chains.sort_by_key(|chain| chain.start);

    let first = chains.remove(0);
    ChainEncoded {
        size: (img.width, img.height),
        fill: first.color,
        chains
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct EdgeInfo<C: Copy + Eq + Debug> {
    color: Option<C>,
    left_is_color: Option<bool>,
    right_is_color: Option<bool>
}

impl<C: Copy + Eq + Debug> EdgeInfo<C> {
    fn unmarked() -> EdgeInfo<C> { EdgeInfo { color: None, left_is_color: None, right_is_color: None }}

    fn mark_color(&mut self, color: C) {
        self.color = Some(color);
    }

    fn mark_as_left_edge(&mut self, color: C) {
        self.color = Some(color);
        self.left_is_color = Some(false);
        if self.right_is_color.is_none() {
            self.right_is_color = Some(true);
        }
    }

    fn mark_as_right_edge(&mut self, color: C) {
        self.color = Some(color);
        self.right_is_color = Some(false);
        if self.left_is_color.is_none() {
            self.left_is_color = Some(true);
        }
    }

    fn mark_as_left_colored(&mut self, color: C) {
        self.color = Some(color);
        self.left_is_color = Some(true);
    }

    fn mark_as_right_colored(&mut self, color: C) {
        self.color = Some(color);
        self.right_is_color = Some(true);
    }
}


fn reconstruct<C: Copy + Eq + Debug>(encoded: ChainEncoded<C>) -> Buffer2D<C> {
    let mut img = Buffer2D::from_fill(encoded.size.0, encoded.size.1, encoded.fill);

    let mut edges = Buffer2D::from_fill(encoded.size.0, encoded.size.1, EdgeInfo::unmarked());
    for chain in encoded.chains {
        if chain.chain.is_empty() {
            edges[chain.start].mark_color(chain.color);
        } else {
            let mut curr = chain.start;
            for offset in chain.chain {
                let next= offset.with(curr).unwrap();
                if offset.y == 1 {
                    // since ccw winding, right is colored
                    if offset.x == 1 {
                        edges[curr].mark_as_right_colored(chain.color);
                        edges[next].mark_as_left_edge(chain.color);
                    } else if offset.x == 0 {
                        edges[curr].mark_as_left_edge(chain.color);
                        edges[next].mark_as_left_edge(chain.color);
                    } else {
                        edges[curr].mark_as_left_edge(chain.color);
                        edges[next].mark_as_right_colored(chain.color);
                    }
                } else if offset.y == -1 {
                    // since ccw, left is colored
                    if offset.x == 1 {
                        edges[curr].mark_as_right_edge(chain.color);
                        edges[next].mark_as_left_colored(chain.color);
                    } else if offset.x == 0 {
                        edges[curr].mark_as_right_edge(chain.color);
                        edges[next].mark_as_right_edge(chain.color);
                    } else {
                        edges[curr].mark_as_left_colored(chain.color);
                        edges[next].mark_as_right_edge(chain.color);
                    }
                } else {
                    // offset.y == 0
                    if offset.x == 1 {
                        edges[curr].mark_as_right_colored(chain.color);
                        edges[next].mark_as_left_colored(chain.color);
                    } else {
                        // offset.x == -1
                        edges[curr].mark_as_left_colored(chain.color);
                        edges[next].mark_as_right_colored(chain.color);
                    }
                }
                curr = next;
            }
        }
    }

    for y in 0..encoded.size.1 {
        let mut curr_color = encoded.fill;
        let mut colors_stack: Vec<C> = Vec::new();
        for x in 0..encoded.size.0 {
            let info = &edges[(x, y)];
            if !info.left_is_color.is_some_and(|v| v) {
                colors_stack.push(curr_color);
            }
            if let Some(color) = info.color {
                img[(x, y)] = color;
                curr_color = color;
            } else {
                img[(x, y)] = curr_color;
            }
            if !info.right_is_color.is_some_and(|v| v) {
                curr_color = colors_stack.pop().unwrap();
            }
        }
    }

    img
}

fn to_image(buffer2d: &Buffer2D<bool>) -> image::GrayImage {
    image::GrayImage::from_fn(buffer2d.width as u32, buffer2d.height as u32, |x, y| [{
        if buffer2d[(x as usize, y as usize)] {
            255
        } else {
            0
        }
    }].into())
}

struct EncodedImage {
    pub name: String,
    pub encoding: ChainEncoded<bool>,
}

impl EncodedImage {
    fn to_bytes<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u16(self.name.len() as u16)?;
        w.write(self.name.as_bytes())?;
        self.encoding.to_bytes(w)?;
        Ok(())
    }

    fn from_bytes<R: Read>(r: &mut R) -> io::Result<EncodedImage> {
        let count = r.read_u16()?;
        let name_buffer = r.read_n(count as usize)?;
        let name = String::from_utf8(name_buffer).unwrap();
        let encoding = ChainEncoded::from_bytes(r)?;
        Ok(EncodedImage {
            name,
            encoding
        })
    }
}

struct Progress {
    state: Arc<AtomicU64>
}

impl Progress {
    fn new(render: impl Fn(u64) -> String + Send + 'static) -> Progress {
        let state = Arc::new(AtomicU64::new(0));
        let state_ref = Arc::downgrade(&state);
        thread::spawn(move || {
            while let Some(state) = state_ref.upgrade() {
                print!("\r{}", render(state.load(Ordering::Relaxed)));
                io::stdout().flush().unwrap();

                thread::sleep(Duration::from_millis(10));
            }
        });

        Progress { state }
    }

    fn tick(&self) {
        self.state.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: Commands
}

#[derive(Subcommand)]
enum Commands {
    Compress {
        dir: PathBuf,
        out: Option<PathBuf>
    },
    Extract {
        file: PathBuf,
        dir: Option<PathBuf>
    }
}

fn compress(dir: PathBuf, out: Option<PathBuf>) {
    let out = out.unwrap_or_else(|| dir.with_extension("min"));

    let mut images = Vec::new();
    for entry in dir.read_dir().unwrap() {
        let entry = entry.unwrap().path();
        if entry.is_file() && entry.extension().is_some_and(|ext| ext == "png") {
            images.push(entry);
        }
    }

    let count = images.len();
    let progress = Progress::new(move |progress| format!("{}/{} images processed", progress, count));

    let mut encoded_images = Vec::new();
    images.into_par_iter().map(|path| {
        let img = image::open(path.clone()).unwrap().to_luma8().view(1, 2, 1440-2, 1080-4).to_image();
        let buffer = Buffer2D::from_fn(img.width() as usize, img.height() as usize, |x, y| img[(x as u32, y as u32)][0] >= 128);
        let encoded = EncodedImage {
            name: path.file_name().unwrap().to_string_lossy().into_owned(),
            encoding: chain_coding( & buffer)
        };

        progress.tick();

        encoded
    }).collect_into_vec(&mut encoded_images);
    drop(progress);

    let file = std::fs::File::create(out).unwrap();
    let encoder = brotlic::BrotliEncoderOptions::new()
        .quality(brotlic::Quality::new(11).unwrap())
        .build().unwrap();
    let mut c = brotlic::CompressorWriter::with_encoder(encoder, file);

    print!("\n");
    let progress = Progress::new(move |progress| format!("{}/{} images written", progress, count));
    c.write_u16(encoded_images.len() as u16).unwrap();
    for image in encoded_images {
        image.to_bytes(&mut c).unwrap();
        progress.tick();
    }
}


fn extract(file: PathBuf, dir: Option<PathBuf>) {
    let dir = dir.unwrap_or_else(|| file.with_extension(""));

    std::fs::create_dir_all(&dir).unwrap();
    let file = std::fs::File::open(&file).unwrap_or_else(|_| {
        panic!("Could not open file {}", file.to_string_lossy());
    });
    let mut reader = brotlic::DecompressorReader::new(BufReader::new(file));

    let count = reader.read_u16().unwrap() as usize;
    let progress = Progress::new(move |progress| format!("{}/{} extracted", progress, count));
    rayon::scope(|s| {
        for _ in 0..count {
            let encoded_image = EncodedImage::from_bytes(&mut reader).unwrap();
            s.spawn(|_| {
                let image = to_image(&reconstruct(encoded_image.encoding));
                let path = dir.join(encoded_image.name);
                image.save(path).unwrap();
                progress.tick();
            });
        }
    });
}


fn main() {
    let cli = Args::parse();

    match cli.command {
        Commands::Compress { dir, out } => compress(dir, out),
        Commands::Extract { file, dir } => extract(file, dir)
    }
}


#[cfg(test)]
mod test {
    use itertools::Itertools;
    use crate::{Buffer2D, chain_coding, ChainEncoded, reconstruct};

    fn test_exhaustive_nxn<const N: usize>() { test_exhaustive_mxn::<N, N>(); }

    fn test_exhaustive_mxn<const M: usize, const N: usize>() {
        for possibility in vec![[true, false]; M * N].into_iter().multi_cartesian_product() {
            let buffer = Buffer2D::from_fn(M, N, |x, y| possibility[y * M + x]);
            let encoded = chain_coding(&buffer);

            let mut byte_buffer = Vec::new();
            encoded.to_bytes(&mut byte_buffer).unwrap();
            let new_buffer = reconstruct(ChainEncoded::from_bytes(&mut byte_buffer.as_slice()).unwrap());

            assert_eq!(buffer, new_buffer);
        }
    }

    #[test]
    fn test_exhaustive_3x3() {
        test_exhaustive_nxn::<3>();
    }

    #[test]
    fn test_exhaustive_3x4() {
        test_exhaustive_mxn::<3, 4>();
    }

    #[test]
    fn test_exhaustive_4x4() {
        test_exhaustive_nxn::<4>();
    }

    #[test]
    fn test_exhaustive_5x4() {
        test_exhaustive_mxn::<5, 4>();
    }
}